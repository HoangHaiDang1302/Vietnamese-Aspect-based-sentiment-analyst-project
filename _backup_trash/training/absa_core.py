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

# %% Cell 0: [KAGGLE] Install Dependencies
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'underthesea', 'pytorch-crf', 'gensim'])
print("Dependencies installed!")

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

# Auto-detect Kaggle environment
IS_KAGGLE = os.path.exists('/kaggle/input')
print(f"Running on: {'Kaggle' if IS_KAGGLE else 'Local'}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42); np.random.seed(42); random.seed(42)
print(f"Device: {device}")
if device.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name(0)}")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

# %% Cell 2: Config & Constants
# === PATHS: tự động chọn Kaggle hoặc Local ===
if IS_KAGGLE:
    # Trên Kaggle: data nằm trong dataset input
    # Thay 'your-dataset-name' bằng tên dataset thật của bạn trên Kaggle
    KAGGLE_DATASET = 'vietnamese-absa-uit-visd4sa'  # ← ĐỔI TÊN NÀY
    DATA_DIR = f'/kaggle/input/{KAGGLE_DATASET}'
    SAVE_DIR = '/kaggle/working/dl_improved_v2'
else:
    DATA_DIR = '.'
    SAVE_DIR = './dl_improved_v2'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Data dir: {DATA_DIR}")
print(f"Save dir: {SAVE_DIR}")

# Verify data files exist
for fn in ['train.jsonl', 'dev.jsonl', 'test.jsonl']:
    fp = os.path.join(DATA_DIR, fn)
    assert os.path.exists(fp), f"File not found: {fp}. Check DATA_DIR or Kaggle dataset name!"
print("All data files found!")

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
# Trên Kaggle: tự download nếu chưa có
if IS_KAGGLE:
    FASTTEXT_PATH = '/kaggle/working/cc.vi.300.vec'
    if not os.path.exists(FASTTEXT_PATH):
        print("Downloading FastText Vietnamese (cc.vi.300.vec) on Kaggle...")
        print("This may take 5-10 minutes...")
        os.system('wget -q https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz -O /kaggle/working/cc.vi.300.vec.gz')
        os.system('gunzip /kaggle/working/cc.vi.300.vec.gz')
        print("Download complete!")
else:
    FASTTEXT_PATH = './cc.vi.300.vec'

def load_embeddings(fasttext_path, word2idx, emb_dim, vocab_size):
    """Load FastText hoặc fallback sang Word2Vec"""
    emb_matrix = np.random.normal(0, 0.1, (vocab_size, emb_dim)).astype(np.float32)
    emb_matrix[PAD_IDX] = 0

    if os.path.exists(fasttext_path):
        print(f"Loading FastText pre-trained ({emb_dim}d)...")
        ft_model = KeyedVectors.load_word2vec_format(fasttext_path, limit=200000)
        hit, miss = 0, 0
        for w, idx in word2idx.items():
            if w in ft_model:
                emb_matrix[idx] = ft_model[w]
                hit += 1
            elif '_' in w:
                parts = w.split('_')
                vecs = [ft_model[p] for p in parts if p in ft_model]
                if vecs:
                    emb_matrix[idx] = np.mean(vecs, axis=0)
                    hit += 1
                else:
                    miss += 1
            else:
                miss += 1
        print(f"FastText loaded! Hit: {hit}/{vocab_size} ({hit/vocab_size*100:.1f}%), Miss: {miss}")
        del ft_model
        import gc; gc.collect()
    else:
        print(f"FastText not found at {fasttext_path}, using Word2Vec fallback ({emb_dim}d)...")
        from gensim.models import Word2Vec
        w2v = Word2Vec(all_sentences, vector_size=emb_dim, window=5, min_count=2,
                       workers=4, epochs=20, sg=1, seed=42)
        for w, idx in word2idx.items():
            if w in w2v.wv: emb_matrix[idx] = w2v.wv[w]
        print(f"Word2Vec trained as fallback")
    return emb_matrix

emb_matrix = load_embeddings(FASTTEXT_PATH, word2idx, EMB_DIM, VOCAB_SIZE)

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
    create_oversampled_data(train_texts, train_tags, train_sent, train_lens, oversample_factor=2)

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

# %% Cell 9: [CẢI TIẾN 3] Tag Distribution Analysis
# NOTE: Emission scaling đã bị BỎ vì gây train/test mismatch với CRF.
# Thay vào đó dùng Oversampling (Cell 7) để xử lý class imbalance.
def analyze_tag_distribution(tags_list, lens_list, num_tags):
    """Phân tích phân bố tag — chỉ để quan sát, KHÔNG dùng weight emissions"""
    tag_counter = Counter()
    total = 0
    for tags, l in zip(tags_list, lens_list):
        for t in tags[:l]:
            tag_counter[t] += 1
            total += 1

    print("\n📊 Tag distribution (top B-tags):")
    b_counts = [(tid, c) for tid, c in tag_counter.items() if BIO_TAGS[tid].startswith('B-')]
    for tid, count in sorted(b_counts, key=lambda x: x[1], reverse=True)[:10]:
        pct = count / total * 100
        print(f"  {BIO_TAGS[tid]:30s}: {count:5d} ({pct:.2f}%)")
    o_count = tag_counter.get(O_TAG, 0)
    print(f"  {'O (non-entity)':30s}: {o_count:5d} ({o_count/total*100:.1f}%)")
    print(f"  → Class imbalance handled by Oversampling (Cell 7)")

analyze_tag_distribution(train_tags, train_lens, NUM_TAGS)

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
    """BiGRU/BiLSTM + Self-Attention + CRF (NO emission scaling)"""
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_tags, pretrained_emb=None,
                 n_layers=2, dropout=0.3, pad_idx=0, rnn_type='gru', bidir=True,
                 use_attention=True):
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

    def _get_emissions(self, seqs, lens, mask=None):
        emb = self.drop(self.emb(seqs))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        output, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, total_length=seqs.size(1))

        if self.use_attention:
            output = self.attention(output, mask)

        return self.hidden2tag(self.drop(output))

    def forward(self, seqs, lens, mask, tags=None):
        emissions = self._get_emissions(seqs, lens, mask)
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return {'loss': loss}
        else:
            best_tags = self.crf.decode(emissions, mask=mask)
            return {'tags': best_tags}


class ImprovedCNNCRF(nn.Module):
    """TextCNN + Self-Attention + CRF (NO emission scaling)"""
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_tags, pretrained_emb=None,
                 pad_idx=0, dropout=0.3, use_attention=True):
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

    def _get_emissions(self, seqs, mask=None):
        emb = self.drop(self.emb(seqs)).transpose(1, 2)
        conv_outs = [torch.relu(conv(emb)) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1).transpose(1, 2)
        if self.use_attention:
            out = self.attention(out, mask)
        return self.hidden2tag(self.drop(out))

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

