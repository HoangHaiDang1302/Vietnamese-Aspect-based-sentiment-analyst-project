import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Bê nguyên bộ Tag 61 Nhãn của bạn vào phục vụ E2E
ASPECTS = ["CAMERA","FEATURES","PERFORMANCE","DESIGN","PRICE",
           "GENERAL","SCREEN","BATTERY","STORAGE","SER&ACC"]
SENTIMENTS = ["POSITIVE","NEUTRAL","NEGATIVE"]
LABEL_NAMES = [f"{a}#{s}" for a in ASPECTS for s in SENTIMENTS]

# Hệ nhãn BIO (Khía Cạnh + Cảm Xúc = Unified Tags)
O_TAG = 0
BIO_TAGS = ['O']
for label in LABEL_NAMES:
    BIO_TAGS.append(f'B-{label}')
    BIO_TAGS.append(f'I-{label}')
TAG2ID = {t:i for i,t in enumerate(BIO_TAGS)}
NUM_TAGS = len(BIO_TAGS)

def text_to_e2e_tags(text, spans):
    """Covert span -> chuỗi tag giống hàm cũ baseline, nhưng lấy mảng raw label dạng word level"""
    words = text.split()
    positions = []
    pos = 0
    text_lower = text.lower()
    
    for w in words:
        w_clean = w.replace('_', ' ')
        idx = text_lower.find(w_clean, pos)
        if idx == -1: idx = pos
        positions.append((idx, idx + len(w_clean)))
        pos = idx + len(w_clean)
        
    tags = ['O'] * len(words)
    sorted_spans = sorted(spans, key=lambda s: s[1]-s[0])
    
    for start_char, end_char, raw_label in sorted_spans:
        b_tag = f'B-{raw_label}'
        i_tag = f'I-{raw_label}'
        if b_tag not in TAG2ID: continue
        
        first_token = True
        for t_idx in range(len(words)):
            t_start, t_end = positions[t_idx]
            if t_start < end_char and t_end > start_char:
                if first_token:
                    tags[t_idx] = b_tag
                    first_token = False
                else:
                    tags[t_idx] = i_tag
    return tags

class E2EDataset(Dataset):
    """
    Dataset Đỉnh Cao: Joint / E2E PhoBERT (Dùng cho cả ATE và ASC chung)
    PhoBERT Tokenizer + Kỹ thuật Re-Alignment chuẩn xác dành cho Subwords
    """
    def __init__(self, items, tokenizer_name="vinai/phobert-base", max_len=128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        
        for item in items:
            text = item['text']
            # Thuật toán convert nhãn cũ theo Word-level
            word_tags = text_to_e2e_tags(text, item.get('labels', []))
            
            # Tokenize theo HuggingFace PhoBERT Subword
            words = text.split()
            # Căn chỉnh nhãn chuẩn xác 100% thủ công (Vì PhoBERT không phải Fast Tokenizer)
            input_ids = [self.tokenizer.cls_token_id]
            aligned_labels = [TAG2ID['O']]  # Thẻ <s> được coi là 'O' để CRF có mask liên tục hoàn hảo

            for i, word in enumerate(words):
                subword_ids = self.tokenizer.encode(word, add_special_tokens=False)
                if not subword_ids: continue
                
                input_ids.extend(subword_ids)
                tag_name = word_tags[i]
                
                # Biến subword liền kề thành 1 cụm I-tag hợp lệ thay vì đục lỗ -100 khiến CRF bị Crash
                for j in range(len(subword_ids)):
                    if j == 0:
                        aligned_labels.append(TAG2ID.get(tag_name, TAG2ID['O']))
                    else:
                        if tag_name.startswith('B-'):
                            aligned_labels.append(TAG2ID.get('I-' + tag_name[2:], TAG2ID['O']))
                        else:
                            aligned_labels.append(TAG2ID.get(tag_name, TAG2ID['O']))

            input_ids.append(self.tokenizer.sep_token_id)
            aligned_labels.append(TAG2ID['O'])  # Thẻ </s>

            # Truncate & Pad
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                aligned_labels = aligned_labels[:max_len]
                attention_mask = [1] * max_len
            else:
                pad_len = max_len - len(input_ids)
                attention_mask = [1] * len(input_ids) + [0] * pad_len
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                aligned_labels = aligned_labels + [-100] * pad_len
                    
            self.input_ids.append(input_ids)
            self.attention_masks.append(attention_mask)
            self.labels.append(aligned_labels)
            
        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.attention_masks = torch.tensor(self.attention_masks, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'tags': self.labels[idx]
        }
