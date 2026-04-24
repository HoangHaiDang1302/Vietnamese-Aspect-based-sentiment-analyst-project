"""
Preprocessing Module
Chứa toàn bộ hàm tiền xử lý: load data, segment, vocab, tokenize, embeddings.
Bao gồm cơ chế save/load để chia sẻ giữa các notebook.
"""
import os
import json
import pickle
from collections import Counter
import numpy as np

try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False

try:
    from gensim.models import Word2Vec
except ImportError:
    pass


# ============================================================
# 1. DATA LOADING
# ============================================================
def load_raw_data(filepath):
    """Đọc file JSONL gốc"""
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items


# ============================================================
# 2. TEXT PREPROCESSING
# ============================================================
def segment_text(text):
    """
    Chuẩn hóa Word Segmentation. Nếu cài underthesea, các từ ghép tiếng Việt sẽ được nối bằng '_'.
    Giúp Mạng nơ-ron học dễ hơn là tách rời từng âm tiết.
    """
    if HAS_UNDERTHESEA:
        try:
            return word_tokenize(text, format="text").lower()
        except:
            return text.lower()
    return text.lower()


def segment_items(items):
    """Segment text cho toàn bộ danh sách items (in-place)."""
    for item in items:
        item['text'] = segment_text(item['text'])
    return items


# ============================================================
# 3. VOCABULARY
# ============================================================
def build_vocab(texts, min_freq=2, special_tokens=None):
    """Tạo Tự điển Vocab. special_tokens: list các token đặc biệt cần thêm (ví dụ ['[ASP]'])"""
    word_freq = Counter(w for t in texts for w in t.split())
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for w, freq in word_freq.items():
        if freq >= min_freq:
            word2idx[w] = len(word2idx)
    # Thêm special tokens nếu có
    if special_tokens:
        for tok in special_tokens:
            if tok not in word2idx:
                word2idx[tok] = len(word2idx)
    return word2idx


def tokenize_baseline(text, word2idx, max_len=128):
    """Chuyển text thành sequence index + padding."""
    words = text.split()[:max_len]
    seq = [word2idx.get(w, 1) for w in words]
    length = max(len(seq), 1)
    seq += [0] * (max_len - len(seq))
    return seq, length


# ============================================================
# 4. WORD2VEC EMBEDDINGS
# ============================================================
def train_w2v_embeddings(texts, word2idx, emb_dim=150, min_freq=2, window_size=5):
    """
    Huấn luyện Word2Vec trực tiếp trên dữ liệu (Self-trained Embeddings).
    Rất hiệu quả cho dữ liệu domain chuyên biệt (review điện thoại).
    """
    all_sentences = [t.split() for t in texts]

    print(f"Training Word2Vec {emb_dim}d...")
    w2v = Word2Vec(all_sentences, vector_size=emb_dim, window=window_size,
                   min_count=min_freq, workers=4, epochs=20, sg=1, seed=42)

    vocab_size = len(word2idx)
    emb_matrix = np.random.normal(0, 0.1, (vocab_size, emb_dim)).astype(np.float32)
    emb_matrix[0] = 0  # PAD

    hit = 0
    for w, idx in word2idx.items():
        if w in w2v.wv:
            emb_matrix[idx] = w2v.wv[w]
            hit += 1

    print(f"Embedding Matrix ready. Hit: {hit}/{vocab_size} ({hit/vocab_size*100:.1f}%)")
    return emb_matrix


# ============================================================
# 5. SAVE / LOAD (Chia sẻ giữa các Notebook)
# ============================================================
def save_preprocessed(save_dir, word2idx, emb_matrix, train_items, dev_items, test_items):
    """
    Lưu toàn bộ vocab, embedding, data đã segment vào thư mục chung.
    Gọi 1 lần duy nhất ở Notebook 01, các notebook sau chỉ cần load.
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'word2idx.pkl'), 'wb') as f:
        pickle.dump(word2idx, f)
    np.save(os.path.join(save_dir, 'emb_matrix.npy'), emb_matrix)

    for name, items in [('train', train_items), ('dev', dev_items), ('test', test_items)]:
        with open(os.path.join(save_dir, f'{name}_segmented.json'), 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False)

    print(f"Saved preprocessed data to {save_dir}/")
    print(f"  - word2idx: {len(word2idx)} tokens")
    print(f"  - emb_matrix: {emb_matrix.shape}")
    print(f"  - train/dev/test: {len(train_items)}/{len(dev_items)}/{len(test_items)}")


def load_preprocessed(save_dir):
    """
    Load vocab, embedding, data đã segment.
    Trả về (word2idx, emb_matrix, train_items, dev_items, test_items).
    """
    with open(os.path.join(save_dir, 'word2idx.pkl'), 'rb') as f:
        word2idx = pickle.load(f)
    emb_matrix = np.load(os.path.join(save_dir, 'emb_matrix.npy'))

    items = {}
    for name in ['train', 'dev', 'test']:
        with open(os.path.join(save_dir, f'{name}_segmented.json'), 'r', encoding='utf-8') as f:
            items[name] = json.load(f)

    print(f"Loaded preprocessed data from {save_dir}/")
    print(f"  - Vocab: {len(word2idx)} | Emb: {emb_matrix.shape}")
    return word2idx, emb_matrix, items['train'], items['dev'], items['test']
