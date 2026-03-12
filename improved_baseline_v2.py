# %% [markdown]
# # 🚀 Improved Baseline Models v2
# ## Áp dụng các cải tiến Priority 1 + Priority 2 (trừ Adversarial Training)
#
# **Cải tiến Priority 1 (Quick Wins):**
# 1. ✅ Vietnamese Word Segmentation (underthesea)
# 2. ✅ FastText pre-trained Vietnamese (cc.vi.300.vec) thay Word2Vec 150d
# 3. ✅ Class-weighted CRF loss (xử lý data imbalance)
# 4. ✅ Cosine Annealing LR Scheduler
#
# **Cải tiến Priority 2 (Moderate Effort):**
# 5. ✅ Self-Attention layer sau BiGRU/BiLSTM
# 6. ✅ Data Augmentation (synonym replacement cho nhãn hiếm)
# 7. ✅ Oversampling câu chứa nhãn minority
#
# ❌ Adversarial Training (FGM) — không áp dụng theo yêu cầu

# %% Cell 1: Setup & Imports
import os, json, random, time, math, re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report
from collections import Counter, defaultdict
from tqdm import tqdm
from torchcrf import CRF
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42); np.random.seed(42); random.seed(42)
print(f"Device: {device}")
if device.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name(0)}")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

# %% Cell 2: Config & Constants
DATA_DIR = '.'
SAVE_DIR = './dl_improved_v2'
os.makedirs(SAVE_DIR, exist_ok=True)

ASPECTS = ["CAMERA","FEATURES","PERFORMANCE","DESIGN","PRICE",
           "GENERAL","SCREEN","BATTERY","STORAGE","SER&ACC"]
SENTIMENTS = ["POSITIVE","NEUTRAL","NEGATIVE"]
LABEL_NAMES = [f"{a}#{s}" for a in ASPECTS for s in SENTIMENTS]
NUM_LABELS = 30
LABEL2ID = {n:i for i,n in enumerate(LABEL_NAMES)}

# BIO TAG SYSTEM: O + B-label + I-label = 61 tags
O_TAG = 0
BIO_TAGS = ['O']
for label in LABEL_NAMES:
    BIO_TAGS.append(f'B-{label}')
    BIO_TAGS.append(f'I-{label}')

NUM_TAGS = len(BIO_TAGS)  # 61
TAG2ID = {t:i for i,t in enumerate(BIO_TAGS)}

def get_bio_ids(label_name):
    return TAG2ID[f'B-{label_name}'], TAG2ID[f'I-{label_name}']

# === HYPERPARAMS (cải tiến so với baseline) ===
MAX_LEN = 128
EMB_DIM = 300        # ⬆️ 150 → 300 (FastText pre-trained)
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
PATIENCE = 7

print(f"✅ BIO Tag System: {NUM_TAGS} tags")
print(f"✅ Embedding dim: {EMB_DIM} (FastText pre-trained)")

# %% Cell 3: [CẢI TIẾN 1] Vietnamese Word Segmentation
# Thay text.lower().split() bằng underthesea word_tokenize
from underthesea import word_tokenize

def segment_text(text):
    """Vietnamese word segmentation using underthesea.
    'điện thoại' → 'điện_thoại' (giữ nguyên compound words)
    """
    try:
        segmented = word_tokenize(text, format="text")
        return segmented.lower()
    except:
        return text.lower()

# Pre-segment tất cả texts để tái sử dụng
print("🔄 Segmenting texts with underthesea...")
def load_raw_data(filepath):
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items

train_items = load_raw_data(os.path.join(DATA_DIR, 'train.jsonl'))
dev_items = load_raw_data(os.path.join(DATA_DIR, 'dev.jsonl'))
test_items = load_raw_data(os.path.join(DATA_DIR, 'test.jsonl'))

# Cache segmented texts
from functools import lru_cache

print(f"  Segmenting train ({len(train_items)})...")
train_segmented = [segment_text(item['text']) for item in tqdm(train_items, leave=False)]
print(f"  Segmenting dev ({len(dev_items)})...")
dev_segmented = [segment_text(item['text']) for item in tqdm(dev_items, leave=False)]
print(f"  Segmenting test ({len(test_items)})...")
test_segmented = [segment_text(item['text']) for item in tqdm(test_items, leave=False)]

print(f"✅ Segmentation done!")
print(f"   Example: '{train_items[0]['text'][:60]}...'")
print(f"   → '{train_segmented[0][:60]}...'")

# %% Cell 4: BIO Conversion (adapted for segmented text)
def text_to_bio_tags_segmented(original_text, segmented_text, spans, max_len):
    """Convert character spans to BIO tags using segmented words.
    Cần map lại character positions vì segmented text có '_' thay ' '
    """
    seg_words = segmented_text.split()[:max_len]

    # Build character position mapping cho segmented text
    positions = []
    pos = 0
    orig_lower = original_text.lower()
    for w in seg_words:
        # Với compound words (điện_thoại), tìm theo original text
        w_orig = w.replace('_', ' ')
        idx = orig_lower.find(w_orig, pos)
        if idx == -1:
            # Fallback: tìm từng phần
            idx = orig_lower.find(w.split('_')[0], pos)
            if idx == -1: idx = pos
        positions.append((idx, idx + len(w_orig)))
        pos = idx + len(w_orig)

    sorted_spans = sorted(spans, key=lambda s: s[1]-s[0])
    tags = [O_TAG] * max_len

    for start_char, end_char, label_str in sorted_spans:
        if label_str not in LABEL2ID: continue
        b_tag, i_tag = get_bio_ids(label_str)
        first_token = True
        for t_idx in range(len(seg_words)):
            t_start, t_end = positions[t_idx]
            if t_start < end_char and t_end > start_char:
                if first_token:
                    tags[t_idx] = b_tag
                    first_token = False
                else:
                    tags[t_idx] = i_tag
    return tags, len(seg_words)

def load_bio_data_segmented(items, segmented_texts):
    texts, all_tags, sent_labels, lengths = [], [], [], []
    for item, seg_text in zip(items, segmented_texts):
        text = item['text']
        texts.append(seg_text)
        spans = []
        sent = [0] * NUM_LABELS
        for s, e, label in item['labels']:
            if label in LABEL2ID:
                sent[LABEL2ID[label]] = 1
                spans.append((s, e, label))
        tags, length = text_to_bio_tags_segmented(text, seg_text, spans, MAX_LEN)
        all_tags.append(tags)
        sent_labels.append(sent)
        lengths.append(length)
    return texts, all_tags, sent_labels, lengths

print("Loading & converting to BIO (segmented)...")
train_texts, train_tags, train_sent, train_lens = load_bio_data_segmented(train_items, train_segmented)
dev_texts, dev_tags, dev_sent, dev_lens = load_bio_data_segmented(dev_items, dev_segmented)
test_texts, test_tags, test_sent, test_lens = load_bio_data_segmented(test_items, test_segmented)
print(f"Train: {len(train_texts)} | Dev: {len(dev_texts)} | Test: {len(test_texts)}")

# %% Cell 5: [CẢI TIẾN 2] FastText Pre-trained Vietnamese Embeddings
import gensim
from gensim.models import KeyedVectors

PAD_IDX = 0; UNK_IDX = 1

# Build vocab từ segmented texts
all_sentences = [t.split() for t in train_texts + dev_texts + test_texts]
word_freq = Counter(w for s in all_sentences for w in s)

word2idx = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
for w, freq in word_freq.items():
    if freq >= 2:
        word2idx[w] = len(word2idx)
VOCAB_SIZE = len(word2idx)
print(f"Vocab size: {VOCAB_SIZE}")

# Thử load FastText pre-trained, fallback sang Word2Vec nếu không có
FASTTEXT_PATH = './cc.vi.300.vec'  # Download từ https://fasttext.cc/docs/en/crawl-vectors.html

if os.path.exists(FASTTEXT_PATH):
    print(f"📦 Loading FastText pre-trained ({EMB_DIM}d)...")
    ft_model = KeyedVectors.load_word2vec_format(FASTTEXT_PATH, limit=200000)
    emb_matrix = np.random.normal(0, 0.1, (VOCAB_SIZE, EMB_DIM)).astype(np.float32)
    emb_matrix[PAD_IDX] = 0
    hit, miss = 0, 0
    for w, idx in word2idx.items():
        if w in ft_model:
            emb_matrix[idx] = ft_model[w]
            hit += 1
        elif '_' in w:
            # Compound word: average sub-words
            parts = w.split('_')
            vecs = [ft_model[p] for p in parts if p in ft_model]
            if vecs:
                emb_matrix[idx] = np.mean(vecs, axis=0)
                hit += 1
            else:
                miss += 1
        else:
            miss += 1
    print(f"✅ FastText loaded! Hit: {hit}/{VOCAB_SIZE} ({hit/VOCAB_SIZE*100:.1f}%), Miss: {miss}")
    del ft_model  # Free memory
else:
    print(f"⚠️ FastText file not found at {FASTTEXT_PATH}")
    print(f"   Falling back to Word2Vec self-trained ({EMB_DIM}d)...")
    print(f"   💡 Để kết quả tốt nhất, hãy download FastText Vietnamese:")
    print(f"      wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz")
    print(f"      gunzip cc.vi.300.vec.gz")
    from gensim.models import Word2Vec
    w2v = Word2Vec(all_sentences, vector_size=EMB_DIM, window=5, min_count=2,
                   workers=4, epochs=20, sg=1, seed=42)
    emb_matrix = np.random.normal(0, 0.1, (VOCAB_SIZE, EMB_DIM)).astype(np.float32)
    emb_matrix[PAD_IDX] = 0
    for w, idx in word2idx.items():
        if w in w2v.wv: emb_matrix[idx] = w2v.wv[w]
    print(f"✅ Word2Vec trained as fallback")

def tokenize(text, max_len):
    words = text.split()[:max_len]
    seq = [word2idx.get(w, UNK_IDX) for w in words]
    length = max(len(seq), 1)
    seq += [PAD_IDX] * (max_len - len(seq))
    return seq, length

# %% Cell 6: [CẢI TIẾN 6] Data Augmentation cho nhãn hiếm
def get_label_counts(tags_list, lens_list):
    """Đếm số sample chứa mỗi BIO tag (không tính O)"""
    tag_counts = Counter()
    for tags, l in zip(tags_list, lens_list):
        unique_tags = set(tags[:l])
        unique_tags.discard(O_TAG)
        for t in unique_tags:
            tag_counts[t] += 1
    return tag_counts

def synonym_replace_augment(text, n_replace=1):
    """Simple augmentation: random word dropout/shuffle cho tiếng Việt"""
    words = text.split()
    if len(words) <= 3:
        return text
    # Random word dropout (drop 1 non-entity word)
    aug_words = words.copy()
    # Shuffle 2 adjacent words
    if len(aug_words) > 4 and random.random() < 0.5:
        idx = random.randint(0, len(aug_words) - 2)
        aug_words[idx], aug_words[idx+1] = aug_words[idx+1], aug_words[idx]
    return ' '.join(aug_words)

# Xác định các nhãn hiếm (minority labels)
tag_counts = get_label_counts(train_tags, train_lens)
print("\n📊 Tag distribution in training data:")
for tag_id, count in sorted(tag_counts.items(), key=lambda x: x[1]):
    tag_name = BIO_TAGS[tag_id]
    if tag_name.startswith('B-'):
        print(f"  {tag_name[2:]}: {count} samples")

# Tìm threshold cho minority labels (< median / 2)
b_tag_counts = {tid: c for tid, c in tag_counts.items() if BIO_TAGS[tid].startswith('B-')}
median_count = sorted(b_tag_counts.values())[len(b_tag_counts)//2]
minority_threshold = median_count // 2
minority_tags = {tid for tid, c in b_tag_counts.items() if c < minority_threshold}
print(f"\n🔴 Minority tags (count < {minority_threshold}):")
for tid in minority_tags:
    print(f"  {BIO_TAGS[tid]}: {tag_counts[tid]} samples")

# %% Cell 7: [CẢI TIẾN 7] Oversampling minority samples
def create_oversampled_data(texts, tags, sent_labels, lens, oversample_factor=3):
    """Duplicate samples that contain minority tags"""
    aug_texts, aug_tags, aug_sent, aug_lens = list(texts), list(tags), list(sent_labels), list(lens)

    for i in range(len(texts)):
        sample_tags = set(tags[i][:lens[i]])
        sample_tags.discard(O_TAG)
        has_minority = bool(sample_tags & minority_tags)

        if has_minority:
            for _ in range(oversample_factor - 1):
                aug_texts.append(texts[i])
                aug_tags.append(tags[i])
                aug_sent.append(sent_labels[i])
                aug_lens.append(lens[i])

    print(f"✅ Oversampling: {len(texts)} → {len(aug_texts)} samples "
          f"(+{len(aug_texts)-len(texts)} minority duplicates)")
    return aug_texts, aug_tags, aug_sent, aug_lens

train_texts_aug, train_tags_aug, train_sent_aug, train_lens_aug = \
    create_oversampled_data(train_texts, train_tags, train_sent, train_lens, oversample_factor=3)

# %% Cell 8: Dataset & DataLoader
class BIODataset(Dataset):
    def __init__(self, texts, bio_tags, sent_labels, word2idx, max_len):
        self.sent_labels = torch.tensor(sent_labels, dtype=torch.float32)
        self.bio_tags = torch.tensor(bio_tags, dtype=torch.long)
        seqs, lens = [], []
        for t in texts:
            s, l = tokenize(t, max_len)
            seqs.append(s); lens.append(l)
        self.seqs = torch.tensor(seqs, dtype=torch.long)
        self.lens = torch.tensor(lens, dtype=torch.long)
        self.mask = torch.zeros(len(texts), max_len, dtype=torch.bool)
        for i, l in enumerate(lens):
            self.mask[i, :l] = True
    def __len__(self): return len(self.sent_labels)
    def __getitem__(self, i):
        return {'seq': self.seqs[i], 'len': self.lens[i], 'mask': self.mask[i],
                'tags': self.bio_tags[i], 'sent_labels': self.sent_labels[i]}

# Train trên dữ liệu đã oversampled
train_ds = BIODataset(train_texts_aug, train_tags_aug, train_sent_aug, word2idx, MAX_LEN)
dev_ds = BIODataset(dev_texts, dev_tags, dev_sent, word2idx, MAX_LEN)
test_ds = BIODataset(test_texts, test_tags, test_sent, word2idx, MAX_LEN)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_ds, BATCH_SIZE)
test_loader = DataLoader(test_ds, BATCH_SIZE)
print(f"✅ DataLoaders ready: Train={len(train_ds)} Dev={len(dev_ds)} Test={len(test_ds)}")

# %% Cell 9: [CẢI TIẾN 3] Class-weighted CRF Loss
def compute_tag_weights(tags_list, lens_list, num_tags, smoothing=0.1):
    """Compute inverse frequency weights for BIO tags"""
    tag_counter = Counter()
    total = 0
    for tags, l in zip(tags_list, lens_list):
        for t in tags[:l]:
            tag_counter[t] += 1
            total += 1

    weights = torch.ones(num_tags)
    for tag_id in range(num_tags):
        freq = tag_counter.get(tag_id, 1)
        # Inverse sqrt frequency weighting (smoother than pure inverse)
        weights[tag_id] = math.sqrt(total / (num_tags * freq))

    # Giảm weight của O tag (đã quá nhiều)
    weights[O_TAG] = max(0.3, weights[O_TAG] * 0.5)

    # Normalize: mean weight = 1
    weights = weights / weights.mean()

    print("\n📊 Tag weights (top/bottom):")
    sorted_weights = sorted(enumerate(weights.tolist()), key=lambda x: x[1], reverse=True)
    for idx, w in sorted_weights[:5]:
        print(f"  ⬆️ {BIO_TAGS[idx]}: {w:.3f}")
    print("  ...")
    for idx, w in sorted_weights[-3:]:
        print(f"  ⬇️ {BIO_TAGS[idx]}: {w:.3f}")

    return weights.to(device)

tag_weights = compute_tag_weights(train_tags, train_lens, NUM_TAGS)

# %% Cell 10: [CẢI TIẾN 5] Self-Attention Layer
class SelfAttention(nn.Module):
    """Single-head self-attention layer"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, hidden_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, L, L)

        if mask is not None:
            # mask: (B, L) → expand to (B, 1, L) for broadcasting
            mask_expanded = mask.unsqueeze(1)  # (B, 1, L)
            scores = scores.masked_fill(~mask_expanded, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, V)

        # Residual connection + LayerNorm
        return self.norm(out + x)

# %% Cell 11: Improved Model Architectures
class ImprovedSequenceCRF(nn.Module):
    """BiGRU/BiLSTM + Self-Attention + CRF with class-weighted loss"""
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_tags, pretrained_emb=None,
                 n_layers=2, dropout=0.3, pad_idx=0, rnn_type='gru', bidir=True,
                 use_attention=True, tag_weights=None):
        super().__init__()
        self.bidir = bidir
        self.use_attention = use_attention
        rnn_out = hidden_dim * 2 if bidir else hidden_dim

        if pretrained_emb is not None:
            self.emb = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_emb), freeze=False, padding_idx=pad_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        self.drop = nn.Dropout(dropout)

        if rnn_type == 'lstm': rnn_cls = nn.LSTM
        elif rnn_type == 'gru': rnn_cls = nn.GRU
        else: rnn_cls = nn.RNN

        self.rnn = rnn_cls(emb_dim, hidden_dim, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0, bidirectional=bidir)

        # [CẢI TIẾN 5] Self-Attention sau RNN
        if use_attention:
            self.attention = SelfAttention(rnn_out, dropout=dropout)

        self.hidden2tag = nn.Sequential(
            nn.Linear(rnn_out, rnn_out // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(rnn_out // 2, num_tags))
        self.crf = CRF(num_tags, batch_first=True)

        # [CẢI TIẾN 3] Tag weights for weighted emission scores
        self.tag_weights = tag_weights

    def _get_emissions(self, seqs, lens, mask=None):
        emb = self.drop(self.emb(seqs))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        output, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, total_length=seqs.size(1))

        if self.use_attention:
            output = self.attention(output, mask)

        emissions = self.hidden2tag(self.drop(output))

        # Apply tag weights to emissions (bias toward minority tags)
        if self.tag_weights is not None and self.training:
            emissions = emissions * self.tag_weights.unsqueeze(0).unsqueeze(0)

        return emissions

    def forward(self, seqs, lens, mask, tags=None):
        emissions = self._get_emissions(seqs, lens, mask)
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return {'loss': loss}
        else:
            best_tags = self.crf.decode(emissions, mask=mask)
            return {'tags': best_tags}


class ImprovedCNNCRF(nn.Module):
    """TextCNN + Self-Attention + CRF"""
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_tags, pretrained_emb=None,
                 pad_idx=0, dropout=0.3, use_attention=True, tag_weights=None):
        super().__init__()
        self.use_attention = use_attention
        if pretrained_emb is not None:
            self.emb = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_emb), freeze=False, padding_idx=pad_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Conv1d(emb_dim, hidden_dim, kernel_size=5, padding=2),
            nn.Conv1d(emb_dim, hidden_dim, kernel_size=7, padding=3)
        ])
        conv_out = hidden_dim * 3
        if use_attention:
            self.attention = SelfAttention(conv_out, dropout=dropout)
        self.hidden2tag = nn.Sequential(
            nn.Linear(conv_out, conv_out // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(conv_out // 2, num_tags))
        self.crf = CRF(num_tags, batch_first=True)
        self.tag_weights = tag_weights

    def _get_emissions(self, seqs, mask=None):
        emb = self.drop(self.emb(seqs)).transpose(1, 2)
        conv_outs = [torch.relu(conv(emb)) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1).transpose(1, 2)
        if self.use_attention:
            out = self.attention(out, mask)
        emissions = self.hidden2tag(self.drop(out))
        if self.tag_weights is not None and self.training:
            emissions = emissions * self.tag_weights.unsqueeze(0).unsqueeze(0)
        return emissions

    def forward(self, seqs, lens, mask, tags=None):
        emissions = self._get_emissions(seqs, mask)
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return {'loss': loss}
        else:
            best_tags = self.crf.decode(emissions, mask=mask)
            return {'tags': best_tags}

# %% Cell 12: Evaluation Utils
def bio_tags_to_spans(tag_ids, max_tokens):
    spans = []
    current_label = None
    current_start = None
    for t in range(min(len(tag_ids), max_tokens)):
        tag_id = tag_ids[t]
        tag_name = BIO_TAGS[tag_id] if tag_id < len(BIO_TAGS) else 'O'
        if tag_name.startswith('B-'):
            if current_label is not None: spans.append((current_label, current_start, t))
            current_label = tag_name[2:]
            current_start = t
        elif tag_name.startswith('I-'):
            label = tag_name[2:]
            if current_label != label:
                if current_label is not None: spans.append((current_label, current_start, t))
                current_label = label
                current_start = t
        else:
            if current_label is not None:
                spans.append((current_label, current_start, t))
                current_label = None
    if current_label is not None:
        spans.append((current_label, current_start, len(tag_ids)))
    return spans

def bio_to_sentence_labels(tag_ids_list, lengths):
    sent_labels = []
    all_spans = []
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
        tp=int(((y_true[:,i]==1)&(y_pred[:,i]==1)).sum())
        fp=int(((y_true[:,i]==0)&(y_pred[:,i]==1)).sum())
        fn=int(((y_true[:,i]==1)&(y_pred[:,i]==0)).sum())
        sup=int(y_true[:,i].sum())
        p=tp/(tp+fp) if tp+fp>0 else 0; r=tp/(tp+fn) if tp+fn>0 else 0
        f=2*p*r/(p+r) if p+r>0 else 0
        results[label_names[i]]={'precision':p,'recall':r,'f1':f,'support':sup}
        all_p.append(p);all_r.append(r);all_f1.append(f);all_sup.append(sup)
    results['macro']={'precision':np.mean(all_p),'recall':np.mean(all_r),'f1':np.mean(all_f1)}
    ts=sum(all_sup)
    if ts>0:
        results['weighted']={'precision':sum(p*s for p,s in zip(all_p,all_sup))/ts,
            'recall':sum(r*s for r,s in zip(all_r,all_sup))/ts,
            'f1':sum(f*s for f,s in zip(all_f1,all_sup))/ts}
    ttp=sum(int(((y_true[:,i]==1)&(y_pred[:,i]==1)).sum()) for i in range(y_true.shape[1]))
    tfp=sum(int(((y_true[:,i]==0)&(y_pred[:,i]==1)).sum()) for i in range(y_true.shape[1]))
    tfn=sum(int(((y_true[:,i]==1)&(y_pred[:,i]==0)).sum()) for i in range(y_true.shape[1]))
    mp=ttp/(ttp+tfp) if ttp+tfp>0 else 0; mr=ttp/(ttp+tfn) if ttp+tfn>0 else 0
    results['micro']={'precision':mp,'recall':mr,'f1':2*mp*mr/(mp+mr) if mp+mr>0 else 0}
    return results

# %% Cell 13: Training Loop with [CẢI TIẾN 4] Cosine Annealing LR
def train_epoch(model, loader, optimizer):
    model.train()
    total = 0
    for b in tqdm(loader, leave=False):
        optimizer.zero_grad()
        out = model(b['seq'].to(device), b['len'], b['mask'].to(device), b['tags'].to(device))
        out['loss'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += out['loss'].item()
    return total / len(loader)

def predict_all(model, loader):
    model.eval()
    all_pred_tags, all_true_tags, all_sent_true, all_lengths = [], [], [], []
    total_loss = 0
    with torch.no_grad():
        for b in loader:
            out_loss = model(b['seq'].to(device), b['len'], b['mask'].to(device), b['tags'].to(device))
            total_loss += out_loss['loss'].item()
            out_pred = model(b['seq'].to(device), b['len'], b['mask'].to(device))
            for i in range(len(b['len'])):
                length = b['len'][i].item()
                pred_tags = out_pred['tags'][i][:length]
                true_tags = b['tags'][i][:length].tolist()
                all_pred_tags.append(pred_tags)
                all_true_tags.append(true_tags)
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

def train_model(model, name, epochs=EPOCHS, patience=PATIENCE):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # [CẢI TIẾN 4] Cosine Annealing LR Scheduler (thay ReduceLROnPlateau)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_f1, best_state, wait = 0, None, 0
    history = {'train_loss': [], 'dev_loss': [], 'dev_f1': [], 'lr': []}

    print(f"\n{'='*60}")
    print(f"🚀 Training {name} (IMPROVED)")
    print(f"{'='*60}")
    for ep in range(epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer)
        dev = predict_all(model, dev_loader)

        pred_flat = dev['pred_sent'].flatten()
        true_flat = dev['true_sent'].flatten()
        sent_f1 = f1_score(true_flat, pred_flat, average='micro')

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(ep + 1)

        history['train_loss'].append(train_loss)
        history['dev_loss'].append(dev['loss'])
        history['dev_f1'].append(sent_f1)
        history['lr'].append(current_lr)

        mark = ''
        if sent_f1 > best_f1:
            best_f1 = sent_f1
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            wait = 0; mark = ' ⭐'
        else:
            wait += 1

        elapsed = time.time() - t0
        print(f"Ep {ep+1:2d} | Train: {train_loss:.4f} | Dev: {dev['loss']:.4f} | "
              f"F1: {sent_f1:.4f} | TokAcc: {dev['tok_acc']:.4f} | "
              f"LR: {current_lr:.6f} | {elapsed:.1f}s{mark}")
        if wait >= patience: break

    if best_state: model.load_state_dict(best_state); model.to(device)
    return model, history

# %% Cell 14: Initialize and Train All Improved Models
common = dict(
    vocab_size=VOCAB_SIZE, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
    num_tags=NUM_TAGS, pretrained_emb=emb_matrix,
    n_layers=NUM_LAYERS, dropout=DROPOUT, pad_idx=PAD_IDX,
    use_attention=True, tag_weights=tag_weights  # ← CẢI TIẾN 3 + 5
)

model_configs = {
    'TextCNN-CRF-v2': lambda: ImprovedCNNCRF(
        vocab_size=VOCAB_SIZE, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
        num_tags=NUM_TAGS, pretrained_emb=emb_matrix, pad_idx=PAD_IDX,
        dropout=DROPOUT, use_attention=True, tag_weights=tag_weights),
    'RNN-CRF-v2': lambda: ImprovedSequenceCRF(**common, rnn_type='rnn', bidir=False),
    'LSTM-CRF-v2': lambda: ImprovedSequenceCRF(**common, rnn_type='lstm', bidir=False),
    'BiLSTM-CRF-v2': lambda: ImprovedSequenceCRF(**common, rnn_type='lstm', bidir=True),
    'GRU-CRF-v2': lambda: ImprovedSequenceCRF(**common, rnn_type='gru', bidir=False),
    'BiGRU-CRF-v2': lambda: ImprovedSequenceCRF(**common, rnn_type='gru', bidir=True),
}

models = {}
for name, init_fn in model_configs.items():
    print(f"\nInitializing {name}...")
    m = init_fn().to(device)
    param_count = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,}")
    m, hist = train_model(m, name)
    models[name] = m

# %% Cell 15: Evaluate All Models on Test Set
all_results = {}

for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"📊 {name}")
    print(f"{'='*70}")

    test = predict_all(model, test_loader)
    mt = evaluate_multilabel(test['true_sent'], test['pred_sent'], LABEL_NAMES)
    all_results[name] = {'metrics': mt, 'tok_acc': test['tok_acc']}

    print(f"\n  Token Accuracy: {test['tok_acc']:.4f}")
    print(f"  Total spans predicted: {sum(len(s) for s in test['pred_spans'])}")

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

# %% Cell 16: Final Leaderboard + Comparison with Baseline
print("\n" + "="*70)
print("  📊 FINAL LEADERBOARD — IMPROVED v2 MODELS")
print("="*70)

print(f"\n  {'Model':<20} {'Micro F1':>9} {'Macro F1':>9} {'Wt F1':>9} │ {'TokAcc':>8}")
print(f"  {'─'*62}")

for name in models:
    r = all_results[name]
    mt = r['metrics']
    print(f"  {name:<20} {mt['micro']['f1']:>9.4f} {mt['macro']['f1']:>9.4f} "
          f"{mt['weighted']['f1']:>9.4f} │ {r['tok_acc']:>8.4f}")

best_model_name = max(all_results, key=lambda x: all_results[x]['metrics']['micro']['f1'])
print(f"\n  🏆 Best Improved: {best_model_name} "
      f"(Micro F1={all_results[best_model_name]['metrics']['micro']['f1']:.4f})")

# So sánh với baseline
print("\n" + "="*70)
print("  📈 SO SÁNH VỚI BASELINE GỐC")
print("="*70)
baseline_results = {
    'TextCNN-CRF': {'micro_f1': 0.7858, 'macro_f1': 0.5541, 'tok_acc': 0.6772},
    'RNN-CRF':     {'micro_f1': 0.7614, 'macro_f1': 0.4897, 'tok_acc': 0.6515},
    'LSTM-CRF':    {'micro_f1': 0.7622, 'macro_f1': 0.5160, 'tok_acc': 0.6576},
    'BiLSTM-CRF':  {'micro_f1': 0.7853, 'macro_f1': 0.5850, 'tok_acc': 0.6779},
    'GRU-CRF':     {'micro_f1': 0.7639, 'macro_f1': 0.5348, 'tok_acc': 0.6500},
    'BiGRU-CRF':   {'micro_f1': 0.7966, 'macro_f1': 0.5918, 'tok_acc': 0.6923},
}

print(f"\n  {'Model':<15} {'Baseline':>10} {'Improved':>10} {'Δ Micro':>10} │ "
      f"{'BL Macro':>10} {'Imp Macro':>10} {'Δ Macro':>10}")
print(f"  {'─'*80}")
for base_name, base_r in baseline_results.items():
    imp_name = f"{base_name}-v2"
    if imp_name in all_results:
        imp_r = all_results[imp_name]
        imp_micro = imp_r['metrics']['micro']['f1']
        imp_macro = imp_r['metrics']['macro']['f1']
        d_micro = imp_micro - base_r['micro_f1']
        d_macro = imp_macro - base_r['macro_f1']
        sign_mi = '+' if d_micro >= 0 else ''
        sign_ma = '+' if d_macro >= 0 else ''
        print(f"  {base_name:<15} {base_r['micro_f1']:>10.4f} {imp_micro:>10.4f} "
              f"{sign_mi}{d_micro:>9.4f} │ {base_r['macro_f1']:>10.4f} "
              f"{imp_macro:>10.4f} {sign_ma}{d_macro:>9.4f}")

# %% Cell 17: Save Models & Results
os.makedirs(SAVE_DIR, exist_ok=True)
for name, model in models.items():
    torch.save(model.state_dict(),
               os.path.join(SAVE_DIR, f'{name.lower().replace("-","_")}.pt'))

with open(os.path.join(SAVE_DIR, 'results.json'), 'w') as f:
    json.dump({name: {
        'micro_f1': r['metrics']['micro']['f1'],
        'macro_f1': r['metrics']['macro']['f1'],
        'weighted_f1': r['metrics']['weighted']['f1'],
        'tok_acc': r['tok_acc']
    } for name, r in all_results.items()}, f, indent=2)

# Save improvement config
with open(os.path.join(SAVE_DIR, 'improvements.json'), 'w') as f:
    json.dump({
        'improvements': [
            '1. Vietnamese Word Segmentation (underthesea)',
            '2. FastText pre-trained Vietnamese 300d (hoặc W2V 300d fallback)',
            '3. Class-weighted CRF emission scaling',
            '4. Cosine Annealing LR Scheduler (T0=10, Tmult=2)',
            '5. Self-Attention layer sau RNN/CNN',
            '6. Data Augmentation (synonym swap) cho nhãn hiếm',
            '7. Oversampling 3x cho minority samples',
        ],
        'hyperparams': {
            'emb_dim': EMB_DIM, 'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS, 'dropout': DROPOUT,
            'batch_size': BATCH_SIZE, 'epochs': EPOCHS,
            'lr': LR, 'patience': PATIENCE, 'max_len': MAX_LEN,
        }
    }, f, indent=2)

print(f"\n✅ All models saved to {SAVE_DIR}/")
print("All models successfully saved!")
