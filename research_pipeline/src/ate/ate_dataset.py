import torch
from torch.utils.data import Dataset
from ..utils.preprocess import tokenize_baseline

# ATE chỉ quan tâm Khía Cạnh, không quan tâm Cảm xúc
# (Ví dụ CAMERA#POSITIVE -> chỉ lấy CAMERA)
ASPECTS = ["CAMERA","FEATURES","PERFORMANCE","DESIGN","PRICE",
           "GENERAL","SCREEN","BATTERY","STORAGE","SER&ACC"]

# B- và I- tags cho ATE
O_TAG = 0
BIO_TAGS = ['O']
for a in ASPECTS:
    BIO_TAGS.append(f'B-{a}')
    BIO_TAGS.append(f'I-{a}')

TAG2ID = {t:i for i,t in enumerate(BIO_TAGS)}
NUM_TAGS = len(BIO_TAGS)

def text_to_ate_tags(text, spans, max_len):
    """Convert spans character -> word BIO level (Chỉ Khía Cạnh)"""
    words = text.split()[:max_len]
    positions = []
    pos = 0
    text_lower = text.lower()
    
    # Tìm kiếm vị trí character offset
    for w in words:
        w_clean = w.replace('_', ' ')  # Phục vụ lỗi nhịp underthesea
        idx = text_lower.find(w_clean, pos)
        if idx == -1: idx = pos
        positions.append((idx, idx + len(w_clean)))
        pos = idx + len(w_clean)
        
    tags = [O_TAG] * max_len
    sorted_spans = sorted(spans, key=lambda s: s[1]-s[0])
    
    for start_char, end_char, raw_label in sorted_spans:
        if '#' not in raw_label: continue
        aspect = raw_label.split('#')[0]
        if f'B-{aspect}' not in TAG2ID: continue
        
        b_id = TAG2ID[f'B-{aspect}']
        i_id = TAG2ID[f'I-{aspect}']
        
        first_token = True
        for t_idx in range(len(words)):
            t_start, t_end = positions[t_idx]
            # Nếu word nằm bên trong span
            if t_start < end_char and t_end > start_char:
                if first_token:
                    tags[t_idx] = b_id
                    first_token = False
                else:
                    tags[t_idx] = i_id
    return tags

class ATEDataset(Dataset):
    """BiODataset chuyên trị Rút trích Khía Cạnh"""
    def __init__(self, items, word2idx, max_len=128):
        self.seqs, self.lens, self.masks, self.tags = [], [], [], []
        
        for item in items:
            text = item['text'] # Giả định Text đã được segment
            spans = item.get('labels', [])
            
            tags = text_to_ate_tags(text, spans, max_len)
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

    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return {'seq': self.seqs[i], 'mask': self.masks[i], 'tags': self.tags[i], 'len': self.lens[i]}
