# %% [markdown]
# # 🔥 PhoBERT-CRF for Vietnamese ABSA
# ## BIO Sequence Labeling with Pre-trained PhoBERT
#
# **Architecture:** PhoBERT-base-v2 → Dropout → Linear → CRF
#
# **Key improvements over RNN baselines:**
# - Contextual embeddings (768d, 12 transformer layers)
# - Vietnamese-specific pre-training (20GB Vietnamese text)
# - Subword tokenization (handles OOV)
# - Differential learning rates

# %% Cell 0: Install Dependencies (Kaggle)
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'py_vncorenlp', 'pytorch-crf', 'transformers'])
print("Dependencies installed!")

# %% Cell 1: Imports & Setup
import os, json, random, time, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from collections import Counter, defaultdict
from tqdm import tqdm
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')

IS_KAGGLE = os.path.exists('/kaggle/input')
print(f"Running on: {'Kaggle' if IS_KAGGLE else 'Local'}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42); np.random.seed(42); random.seed(42)
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# %% Cell 2: Config
if IS_KAGGLE:
    KAGGLE_DATASET = 'vietnamese-absa-uit-visd4sa'  # ← ĐỔI TÊN NÀY
    DATA_DIR = f'/kaggle/input/{KAGGLE_DATASET}'
    SAVE_DIR = '/kaggle/working/phobert_crf'
else:
    DATA_DIR = '.'
    SAVE_DIR = './phobert_crf'
os.makedirs(SAVE_DIR, exist_ok=True)

# Verify data
for fn in ['train.jsonl', 'dev.jsonl', 'test.jsonl']:
    assert os.path.exists(os.path.join(DATA_DIR, fn)), f"Not found: {fn}"
print(f"Data: {DATA_DIR}\nSave: {SAVE_DIR}")

# === Label System ===
ASPECTS = ["CAMERA","FEATURES","PERFORMANCE","DESIGN","PRICE",
           "GENERAL","SCREEN","BATTERY","STORAGE","SER&ACC"]
SENTIMENTS = ["POSITIVE","NEUTRAL","NEGATIVE"]
LABEL_NAMES = [f"{a}#{s}" for a in ASPECTS for s in SENTIMENTS]
NUM_LABELS = 30
LABEL2ID = {n:i for i,n in enumerate(LABEL_NAMES)}

O_TAG = 0
BIO_TAGS = ['O']
for label in LABEL_NAMES:
    BIO_TAGS.append(f'B-{label}')
    BIO_TAGS.append(f'I-{label}')
NUM_TAGS = len(BIO_TAGS)  # 61
TAG2ID = {t:i for i,t in enumerate(BIO_TAGS)}

def get_bio_ids(label_name):
    return TAG2ID[f'B-{label_name}'], TAG2ID[f'I-{label_name}']

# === Hyperparameters ===
PHOBERT_MODEL = "vinai/phobert-base-v2"
MAX_LEN = 256       # subword tokens (PhoBERT max = 256)
BATCH_SIZE = 16
GRAD_ACCUM = 2      # effective batch = 32
EPOCHS = 15
LR_BERT = 2e-5      # PhoBERT layers (fine-tune nhẹ)
LR_HEAD = 1e-3      # Classifier + CRF (train từ đầu)
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DROPOUT = 0.1
PATIENCE = 5

print(f"✅ Config: {NUM_TAGS} BIO tags, model={PHOBERT_MODEL}")

# %% Cell 3: Load PhoBERT Tokenizer + Word Segmentation
import py_vncorenlp
import os

# Download VnCoreNLP model if it doesn't exist
vncorenlp_dir = os.path.abspath('./VnCoreNLP')
if not os.path.exists(vncorenlp_dir):
    os.makedirs(vncorenlp_dir, exist_ok=True)
    py_vncorenlp.download_model(save_dir=vncorenlp_dir)

# Load rdrsegmenter
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_dir)

tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)
print(f"✅ PhoBERT tokenizer loaded (vocab={tokenizer.vocab_size})")

def segment_text(text):
    """PhoBERT yêu cầu text đã word-segment"""
    try:
        sentences = rdrsegmenter.word_segment(text)
        return " ".join(sentences)
    except:
        return text

# %% Cell 4: Load & Prepare Data
def load_raw_data(filepath):
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items

print("Loading data...")
train_items = load_raw_data(os.path.join(DATA_DIR, 'train.jsonl'))
dev_items = load_raw_data(os.path.join(DATA_DIR, 'dev.jsonl'))
test_items = load_raw_data(os.path.join(DATA_DIR, 'test.jsonl'))
print(f"Train: {len(train_items)} | Dev: {len(dev_items)} | Test: {len(test_items)}")

# Segment all texts
print("Segmenting texts with underthesea...")
for items in [train_items, dev_items, test_items]:
    for item in tqdm(items, leave=False):
        item['segmented'] = segment_text(item['text'])

print(f"Example: '{train_items[0]['text'][:60]}...'")
print(f"    Seg: '{train_items[0]['segmented'][:60]}...'")

# %% Cell 5: PhoBERT Tokenization + BIO Alignment (CORE)
def tokenize_and_align(item, tokenizer, max_len):
    """
    Core function: tokenize segmented text with PhoBERT,
    then align character-level spans to subword tokens.

    Returns: input_ids, attention_mask, word_ids, bio_tags, word_count
    """
    seg_text = item['segmented']
    original_text = item['text']

    # Step 1: Split segmented text into words
    words = seg_text.split()

    # Step 2: PhoBERT tokenize word by word, tracking word boundaries
    all_input_ids = [tokenizer.cls_token_id]  # [CLS]
    word_ids_map = [-1]  # -1 for special tokens
    for word_idx, word in enumerate(words):
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(all_input_ids) + len(tokens) + 1 > max_len:  # +1 for [SEP]
            break
        all_input_ids.extend(tokens)
        word_ids_map.extend([word_idx] * len(tokens))

    all_input_ids.append(tokenizer.sep_token_id)  # [SEP]
    word_ids_map.append(-1)

    seq_len = len(all_input_ids)
    attention_mask = [1] * seq_len

    # Pad to max_len
    pad_len = max_len - seq_len
    all_input_ids += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len
    word_ids_map += [-1] * pad_len

    # Step 3: Compute word-level character positions (from ORIGINAL text)
    word_count = max(word_ids_map) + 1 if word_ids_map else 0
    positions = []
    pos = 0
    orig_lower = original_text.lower()
    for w in words[:word_count]:
        w_orig = w.replace('_', ' ')
        idx = orig_lower.find(w_orig, pos)
        if idx == -1:
            idx = orig_lower.find(w.split('_')[0], pos)
            if idx == -1: idx = pos
        positions.append((idx, idx + len(w_orig)))
        pos = idx + len(w_orig)

    # Step 4: Assign BIO tags at WORD level
    word_tags = [O_TAG] * word_count
    spans = [(s, e, l) for s, e, l in item['labels'] if l in LABEL2ID]
    sorted_spans = sorted(spans, key=lambda s: s[1] - s[0])

    for start_char, end_char, label_str in sorted_spans:
        b_tag, i_tag = get_bio_ids(label_str)
        first_token = True
        for w_idx in range(word_count):
            w_start, w_end = positions[w_idx]
            if w_start < end_char and w_end > start_char:
                if first_token:
                    word_tags[w_idx] = b_tag
                    first_token = False
                else:
                    word_tags[w_idx] = i_tag

    # Step 5: Sentence-level labels
    sent_labels = [0] * NUM_LABELS
    for _, _, label in spans:
        sent_labels[LABEL2ID[label]] = 1

    return {
        'input_ids': all_input_ids,
        'attention_mask': attention_mask,
        'word_ids': word_ids_map,
        'word_tags': word_tags,       # length = word_count
        'word_count': word_count,
        'sent_labels': sent_labels,
    }

# Test
sample = tokenize_and_align(train_items[0], tokenizer, MAX_LEN)
print(f"Sample: {sample['word_count']} words, {sum(sample['attention_mask'])} subwords")
print(f"Tags: {[BIO_TAGS[t] for t in sample['word_tags'] if t != 0]}")

# %% Cell 6: Oversampling Minority Samples
def find_minority_tags(items, tokenizer, max_len):
    """Find tags with few samples"""
    tag_counts = Counter()
    for item in items:
        data = tokenize_and_align(item, tokenizer, max_len)
        for t in data['word_tags']:
            if t != O_TAG:
                tag_counts[t] += 1

    b_counts = {t: c for t, c in tag_counts.items() if BIO_TAGS[t].startswith('B-')}
    if not b_counts:
        return set()
    median = sorted(b_counts.values())[len(b_counts) // 2]
    threshold = median // 2
    minority = {t for t, c in b_counts.items() if c < threshold}
    print(f"\nMinority tags (count < {threshold}):")
    for t in minority:
        print(f"  {BIO_TAGS[t]}: {tag_counts[t]}")
    return minority

print("Analyzing tag distribution...")
minority_tags = find_minority_tags(train_items, tokenizer, MAX_LEN)

# Oversample 2x
train_items_aug = list(train_items)
added = 0
for item in train_items:
    data = tokenize_and_align(item, tokenizer, MAX_LEN)
    has_minority = any(t in minority_tags for t in data['word_tags'])
    if has_minority:
        train_items_aug.append(item)
        added += 1
print(f"✅ Oversampling: {len(train_items)} → {len(train_items_aug)} (+{added} minority)")

# %% Cell 7: Dataset
class PhoBERTBIODataset(Dataset):
    def __init__(self, items, tokenizer, max_len):
        self.data = []
        for item in tqdm(items, desc="Tokenizing", leave=False):
            self.data.append(tokenize_and_align(item, tokenizer, max_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            'input_ids': torch.tensor(d['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(d['attention_mask'], dtype=torch.long),
            'word_ids': torch.tensor(d['word_ids'], dtype=torch.long),
            'word_count': d['word_count'],
            'word_tags': d['word_tags'],  # variable length, pad in collate
            'sent_labels': torch.tensor(d['sent_labels'], dtype=torch.float32),
        }

def collate_fn(batch):
    """Custom collate: pad word_tags to max word_count in batch"""
    max_words = max(b['word_count'] for b in batch)

    word_tags_padded = []
    for b in batch:
        tags = list(b['word_tags'])  # COPY to avoid mutating dataset!
        tags = tags + [O_TAG] * (max_words - len(tags))
        word_tags_padded.append(tags)

    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'word_ids': torch.stack([b['word_ids'] for b in batch]),
        'word_count': torch.tensor([b['word_count'] for b in batch], dtype=torch.long),
        'word_tags': torch.tensor(word_tags_padded, dtype=torch.long),
        'sent_labels': torch.stack([b['sent_labels'] for b in batch]),
    }

print("Building datasets...")
train_ds = PhoBERTBIODataset(train_items_aug, tokenizer, MAX_LEN)
dev_ds = PhoBERTBIODataset(dev_items, tokenizer, MAX_LEN)
test_ds = PhoBERTBIODataset(test_items, tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, BATCH_SIZE, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, BATCH_SIZE, collate_fn=collate_fn)
print(f"✅ DataLoaders: Train={len(train_ds)} Dev={len(dev_ds)} Test={len(test_ds)}")

# %% Cell 8: PhoBERT-CRF Model
class PhoBERTCRF(nn.Module):
    """PhoBERT + Linear + CRF for BIO Sequence Labeling"""

    def __init__(self, model_name, num_tags, dropout=0.1):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.phobert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def _get_word_emissions(self, input_ids, attention_mask, word_ids, word_count):
        """
        Run PhoBERT, then extract FIRST subword representation per word.
        Returns: emissions (batch, max_words, num_tags), word_mask (batch, max_words)
        """
        # PhoBERT forward
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)  # (B, seq_len, 768)

        batch_size = input_ids.size(0)
        max_words = word_count.max().item()

        # Extract first-subword representation per word
        word_emissions = torch.zeros(batch_size, max_words, self.hidden_size,
                                     device=input_ids.device)
        word_mask = torch.zeros(batch_size, max_words, dtype=torch.bool,
                                device=input_ids.device)

        for b in range(batch_size):
            wc = word_count[b].item()
            word_mask[b, :wc] = True
            seen_words = set()
            for pos in range(word_ids.size(1)):
                wid = word_ids[b, pos].item()
                if wid >= 0 and wid < wc and wid not in seen_words:
                    word_emissions[b, wid] = sequence_output[b, pos]
                    seen_words.add(wid)

        emissions = self.classifier(word_emissions)  # (B, max_words, num_tags)
        return emissions, word_mask

    def forward(self, input_ids, attention_mask, word_ids, word_count, word_tags=None):
        emissions, word_mask = self._get_word_emissions(
            input_ids, attention_mask, word_ids, word_count)

        if word_tags is not None:
            # Truncate tags to match max_words
            max_words = word_count.max().item()
            tags = word_tags[:, :max_words]
            loss = -self.crf(emissions, tags, mask=word_mask, reduction='mean')
            return {'loss': loss}
        else:
            best_tags = self.crf.decode(emissions, mask=word_mask)
            return {'tags': best_tags}

# Initialize model
model = PhoBERTCRF(PHOBERT_MODEL, NUM_TAGS, DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ PhoBERT-CRF initialized")
print(f"   Total params: {total_params:,}")
print(f"   Trainable: {trainable_params:,}")

# %% Cell 9: Evaluation Utils
def bio_tags_to_spans(tag_ids, max_tokens):
    spans = []
    current_label, current_start = None, None
    for t in range(min(len(tag_ids), max_tokens)):
        tag_id = tag_ids[t]
        tag_name = BIO_TAGS[tag_id] if tag_id < len(BIO_TAGS) else 'O'
        if tag_name.startswith('B-'):
            if current_label is not None: spans.append((current_label, current_start, t))
            current_label = tag_name[2:]; current_start = t
        elif tag_name.startswith('I-'):
            label = tag_name[2:]
            if current_label != label:
                if current_label is not None: spans.append((current_label, current_start, t))
                current_label = label; current_start = t
        else:
            if current_label is not None:
                spans.append((current_label, current_start, t))
                current_label = None
    if current_label is not None:
        spans.append((current_label, current_start, len(tag_ids)))
    return spans

def bio_to_sentence_labels(tag_ids_list, lengths):
    sent_labels, all_spans = [], []
    for tag_ids, length in zip(tag_ids_list, lengths):
        spans = bio_tags_to_spans(tag_ids, length)
        label_vec = [0] * NUM_LABELS
        for label_name, _, _ in spans:
            if label_name in LABEL2ID:
                label_vec[LABEL2ID[label_name]] = 1
        sent_labels.append(label_vec)
        all_spans.append(spans)
    return np.array(sent_labels), all_spans

def evaluate_multilabel(y_true, y_pred, label_names):
    y_true, y_pred = np.array(y_true).astype(int), np.array(y_pred).astype(int)
    results = {}
    all_p, all_r, all_f1, all_sup = [], [], [], []
    for i in range(y_true.shape[1]):
        tp = int(((y_true[:,i]==1)&(y_pred[:,i]==1)).sum())
        fp = int(((y_true[:,i]==0)&(y_pred[:,i]==1)).sum())
        fn = int(((y_true[:,i]==1)&(y_pred[:,i]==0)).sum())
        sup = int(y_true[:,i].sum())
        p = tp/(tp+fp) if tp+fp>0 else 0
        r = tp/(tp+fn) if tp+fn>0 else 0
        f = 2*p*r/(p+r) if p+r>0 else 0
        results[label_names[i]] = {'precision':p,'recall':r,'f1':f,'support':sup}
        all_p.append(p); all_r.append(r); all_f1.append(f); all_sup.append(sup)
    results['macro'] = {'precision':np.mean(all_p),'recall':np.mean(all_r),'f1':np.mean(all_f1)}
    ts = sum(all_sup)
    if ts>0:
        results['weighted'] = {
            'precision':sum(p*s for p,s in zip(all_p,all_sup))/ts,
            'recall':sum(r*s for r,s in zip(all_r,all_sup))/ts,
            'f1':sum(f*s for f,s in zip(all_f1,all_sup))/ts}
    ttp = sum(int(((y_true[:,i]==1)&(y_pred[:,i]==1)).sum()) for i in range(y_true.shape[1]))
    tfp = sum(int(((y_true[:,i]==0)&(y_pred[:,i]==1)).sum()) for i in range(y_true.shape[1]))
    tfn = sum(int(((y_true[:,i]==1)&(y_pred[:,i]==0)).sum()) for i in range(y_true.shape[1]))
    mp = ttp/(ttp+tfp) if ttp+tfp>0 else 0
    mr = ttp/(ttp+tfn) if ttp+tfn>0 else 0
    results['micro'] = {'precision':mp,'recall':mr,'f1':2*mp*mr/(mp+mr) if mp+mr>0 else 0}
    return results

# %% Cell 10: Training Loop with Differential LR
def predict_all(model, loader):
    model.eval()
    all_pred_tags, all_true_tags, all_sent_true, all_lengths = [], [], [], []
    total_loss = 0

    with torch.no_grad():
        for b in loader:
            ids = b['input_ids'].to(device)
            mask = b['attention_mask'].to(device)
            wids = b['word_ids'].to(device)
            wc = b['word_count'].to(device)
            wtags = b['word_tags'].to(device)

            out_loss = model(ids, mask, wids, wc, wtags)
            total_loss += out_loss['loss'].item()

            out_pred = model(ids, mask, wids, wc)
            for i in range(len(b['word_count'])):
                length = b['word_count'][i].item()
                pred = out_pred['tags'][i][:length]
                true = b['word_tags'][i][:length].tolist()
                all_pred_tags.append(pred)
                all_true_tags.append(true)
                all_lengths.append(length)
            all_sent_true.extend(b['sent_labels'].numpy())

    pred_sent, pred_spans = bio_to_sentence_labels(all_pred_tags, all_lengths)
    true_sent, true_spans = bio_to_sentence_labels(all_true_tags, all_lengths)

    correct, total_tokens = 0, 0
    for pt, tt in zip(all_pred_tags, all_true_tags):
        for p, t in zip(pt, tt):
            if p == t: correct += 1
            total_tokens += 1
    tok_acc = correct / max(1, total_tokens)

    return {'loss': total_loss / len(loader), 'pred_sent': pred_sent,
            'true_sent': np.array(all_sent_true), 'pred_spans': pred_spans,
            'true_spans': true_spans, 'tok_acc': tok_acc}


def train_phobert():
    # Differential learning rates
    bert_params = list(model.phobert.parameters())
    head_params = list(model.classifier.parameters()) + list(model.crf.parameters())

    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': LR_BERT, 'weight_decay': WEIGHT_DECAY},
        {'params': head_params, 'lr': LR_HEAD, 'weight_decay': 0.0},
    ])

    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1, best_state, wait = 0, None, 0
    history = {'train_loss': [], 'dev_loss': [], 'dev_f1': [], 'tok_acc': []}

    print(f"\n{'='*70}")
    print(f"🔥 Training PhoBERT-CRF")
    print(f"   BERT LR: {LR_BERT}, Head LR: {LR_HEAD}")
    print(f"   Batch: {BATCH_SIZE} x {GRAD_ACCUM} accum = {BATCH_SIZE*GRAD_ACCUM} effective")
    print(f"   Total steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"{'='*70}")

    for ep in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, b in enumerate(tqdm(train_loader, desc=f"Ep {ep+1}", leave=False)):
            ids = b['input_ids'].to(device)
            mask = b['attention_mask'].to(device)
            wids = b['word_ids'].to(device)
            wc = b['word_count'].to(device)
            wtags = b['word_tags'].to(device)

            out = model(ids, mask, wids, wc, wtags)
            loss = out['loss'] / GRAD_ACCUM
            loss.backward()
            total_loss += out['loss'].item()

            if (step + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Handle remaining gradients
        if (step + 1) % GRAD_ACCUM != 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)

        # Evaluate on dev
        dev = predict_all(model, dev_loader)
        pred_flat = dev['pred_sent'].flatten()
        true_flat = dev['true_sent'].flatten()
        sent_f1 = f1_score(true_flat, pred_flat, average='micro')

        history['train_loss'].append(avg_train_loss)
        history['dev_loss'].append(dev['loss'])
        history['dev_f1'].append(sent_f1)
        history['tok_acc'].append(dev['tok_acc'])

        current_lr = scheduler.get_last_lr()[0]
        mark = ''
        if sent_f1 > best_f1:
            best_f1 = sent_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0; mark = ' ⭐'
        else:
            wait += 1

        print(f"Ep {ep+1:2d} | Train: {avg_train_loss:.4f} | Dev: {dev['loss']:.4f} | "
              f"F1: {sent_f1:.4f} | TokAcc: {dev['tok_acc']:.4f} | "
              f"LR: {current_lr:.2e}{mark}")

        if wait >= PATIENCE:
            print(f"Early stopping at epoch {ep+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    return history

history = train_phobert()

# %% Cell 11: Evaluate on Test Set
print(f"\n{'='*70}")
print(f"📊 PhoBERT-CRF — Test Set Evaluation")
print(f"{'='*70}")

test_out = predict_all(model, test_loader)
mt = evaluate_multilabel(test_out['true_sent'], test_out['pred_sent'], LABEL_NAMES)

print(f"\n  Token Accuracy: {test_out['tok_acc']:.4f}")
print(f"  Total spans predicted: {sum(len(s) for s in test_out['pred_spans'])}")

print(f"\n  {'Avg':<15} {'P':>8} {'R':>8} {'F1':>8}")
print(f"  {'-'*42}")
for avg in ['micro', 'macro', 'weighted']:
    m = mt[avg]
    print(f"  {avg:<15} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")

print(f"\n  {'Label':<25} {'P':>7} {'R':>7} {'F1':>7} {'Sup':>6}")
print(f"  {'-'*55}")
for ln in LABEL_NAMES:
    m = mt[ln]
    flag = '🔴' if m['support'] < 20 else ''
    print(f"  {ln:<25} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['support']:>6d} {flag}")

# %% Cell 12: Compare with Baselines
print(f"\n{'='*70}")
print(f"📈 SO SÁNH: PhoBERT-CRF vs Baselines")
print(f"{'='*70}")

baseline_results = {
    'TextCNN-CRF':  {'micro_f1': 0.7858, 'macro_f1': 0.5541},
    'RNN-CRF':      {'micro_f1': 0.7614, 'macro_f1': 0.4897},
    'LSTM-CRF':     {'micro_f1': 0.7622, 'macro_f1': 0.5160},
    'BiLSTM-CRF':   {'micro_f1': 0.7853, 'macro_f1': 0.5850},
    'GRU-CRF':      {'micro_f1': 0.7639, 'macro_f1': 0.5348},
    'BiGRU-CRF':    {'micro_f1': 0.7966, 'macro_f1': 0.5918},
}

phobert_micro = mt['micro']['f1']
phobert_macro = mt['macro']['f1']

print(f"\n  {'Model':<20} {'Micro F1':>10} {'Macro F1':>10}")
print(f"  {'─'*45}")
for name, r in baseline_results.items():
    print(f"  {name:<20} {r['micro_f1']:>10.4f} {r['macro_f1']:>10.4f}")
print(f"  {'─'*45}")
print(f"  {'PhoBERT-CRF':<20} {phobert_micro:>10.4f} {phobert_macro:>10.4f}  🔥")

best_baseline_micro = max(r['micro_f1'] for r in baseline_results.values())
delta = phobert_micro - best_baseline_micro
print(f"\n  Δ vs best baseline (BiGRU-CRF): {'+' if delta >= 0 else ''}{delta:.4f} Micro F1")

# %% Cell 13: Save Model & Results
# Save model
torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'phobert_crf.pt'))

# Save results
results = {
    'model': 'PhoBERT-CRF',
    'phobert_model': PHOBERT_MODEL,
    'micro_f1': mt['micro']['f1'],
    'macro_f1': mt['macro']['f1'],
    'weighted_f1': mt['weighted']['f1'],
    'tok_acc': test_out['tok_acc'],
    'hyperparams': {
        'max_len': MAX_LEN, 'batch_size': BATCH_SIZE,
        'grad_accum': GRAD_ACCUM, 'epochs': EPOCHS,
        'lr_bert': LR_BERT, 'lr_head': LR_HEAD,
        'dropout': DROPOUT, 'patience': PATIENCE,
    },
    'per_label': {ln: mt[ln] for ln in LABEL_NAMES},
}

with open(os.path.join(SAVE_DIR, 'results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Model & results saved to {SAVE_DIR}/")
print(f"   phobert_crf.pt ({os.path.getsize(os.path.join(SAVE_DIR, 'phobert_crf.pt'))/1024/1024:.1f} MB)")
print("Done!")
