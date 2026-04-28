"""
E2E Baseline Dataset - End-to-End ABSA dùng word2idx (không dùng PhoBERT)
Sử dụng Unified Tags giống E2E PhoBERT (B-CAMERA#POSITIVE, I-CAMERA#POSITIVE, ...)
nhưng tokenize bằng word2idx + Word2Vec, giúp so sánh công bằng giữa:
  - Pipeline (BiGRU-CRF + BiGRU)
  - E2E BiGRU-CRF (Unified Tags)  <-- baseline này
  - E2E PhoBERT-CRF (Unified Tags)
"""
import torch
from torch.utils.data import Dataset
from ..utils.preprocess import tokenize_baseline

# Reuse Unified Tags từ E2E PhoBERT
ASPECTS = ["CAMERA","FEATURES","PERFORMANCE","DESIGN","PRICE",
           "GENERAL","SCREEN","BATTERY","STORAGE","SER&ACC"]
SENTIMENTS = ["POSITIVE","NEUTRAL","NEGATIVE"]
LABEL_NAMES = [f"{a}#{s}" for a in ASPECTS for s in SENTIMENTS]

O_TAG = 0
BIO_TAGS = ['O']
for label in LABEL_NAMES:
    BIO_TAGS.append(f'B-{label}')
    BIO_TAGS.append(f'I-{label}')

TAG2ID = {t: i for i, t in enumerate(BIO_TAGS)}
NUM_TAGS = len(BIO_TAGS)


def text_to_unified_tags(text, spans, max_len):
    """
    Convert character-level spans → word-level Unified BIO tags.
    Giống text_to_ate_tags nhưng dùng FULL label (CATEGORY#SENTIMENT)
    thay vì chỉ CATEGORY.
    """
    words = text.split()[:max_len]
    positions = []
    pos = 0
    text_lower = text.lower()

    for w in words:
        w_clean = w.replace('_', ' ')
        idx = text_lower.find(w_clean, pos)
        if idx == -1:
            idx = pos
        positions.append((idx, idx + len(w_clean)))
        pos = idx + len(w_clean)

    tags = [O_TAG] * max_len
    sorted_spans = sorted(spans, key=lambda s: s[1] - s[0])

    for start_char, end_char, raw_label in sorted_spans:
        # raw_label dạng "CAMERA#POSITIVE"
        b_tag_name = f'B-{raw_label}'
        i_tag_name = f'I-{raw_label}'
        if b_tag_name not in TAG2ID:
            continue

        b_id = TAG2ID[b_tag_name]
        i_id = TAG2ID[i_tag_name]

        first_token = True
        for t_idx in range(len(words)):
            t_start, t_end = positions[t_idx]
            if t_start < end_char and t_end > start_char:
                if first_token:
                    tags[t_idx] = b_id
                    first_token = False
                else:
                    tags[t_idx] = i_id

    return tags


class E2EBaselineDataset(Dataset):
    """
    E2E Dataset dùng word2idx tokenization (không dùng PhoBERT).
    Output format giống ATEDataset nhưng tags là Unified Tags.
    """
    def __init__(self, items, word2idx, max_len=128):
        self.seqs, self.lens, self.masks, self.tags = [], [], [], []

        for item in items:
            text = item['text']  # Giả định đã được segment
            spans = item.get('labels', [])

            tags = text_to_unified_tags(text, spans, max_len)
            seq, length = tokenize_baseline(text, word2idx, max_len)
            mask = [1] * length + [0] * (max_len - length)

            self.seqs.append(seq)
            self.lens.append(length)
            self.masks.append(mask)
            self.tags.append(tags)

        self.seqs = torch.tensor(self.seqs, dtype=torch.long)
        self.lens = torch.tensor(self.lens, dtype=torch.long)
        self.masks = torch.tensor(self.masks, dtype=torch.bool)
        self.tags = torch.tensor(self.tags, dtype=torch.long)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return {
            'seq': self.seqs[i],
            'mask': self.masks[i],
            'tags': self.tags[i],
            'len': self.lens[i]
        }
