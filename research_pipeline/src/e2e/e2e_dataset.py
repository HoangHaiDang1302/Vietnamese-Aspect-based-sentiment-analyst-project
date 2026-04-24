import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
from collections import Counter
import py_vncorenlp

# Download VnCoreNLP model if it doesn't exist
vncorenlp_dir = os.path.abspath('./VnCoreNLP')
if not os.path.exists(vncorenlp_dir):
    os.makedirs(vncorenlp_dir, exist_ok=True)
    py_vncorenlp.download_model(save_dir=vncorenlp_dir)

# Initialize rdrsegmenter globally to avoid reloading
try:
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_dir)
except:
    rdrsegmenter = None

# Unified Tags System
ASPECTS = ["CAMERA","FEATURES","PERFORMANCE","DESIGN","PRICE",
           "GENERAL","SCREEN","BATTERY","STORAGE","SER&ACC"]
SENTIMENTS = ["POSITIVE","NEUTRAL","NEGATIVE"]
LABEL_NAMES = [f"{a}#{s}" for a in ASPECTS for s in SENTIMENTS]

O_TAG = 0
BIO_TAGS = ['O']
for label in LABEL_NAMES:
    BIO_TAGS.append(f'B-{label}')
    BIO_TAGS.append(f'I-{label}')
TAG2ID = {t:i for i,t in enumerate(BIO_TAGS)}
NUM_TAGS = len(BIO_TAGS)

def segment_text(text):
    if rdrsegmenter is None:
        return text
    try:
        sentences = rdrsegmenter.word_segment(text)
        return " ".join(sentences)
    except:
        return text

def tokenize_and_align(item, tokenizer, max_len):
    """Tokenize segmented text with PhoBERT, align character-level spans to word tokens."""
    original_text = item['text']
    seg_text = segment_text(original_text)
    words = seg_text.split()

    all_input_ids = [tokenizer.cls_token_id]
    word_ids_map = [-1] 

    for word_idx, word in enumerate(words):
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(all_input_ids) + len(tokens) + 1 > max_len:
            break
        all_input_ids.extend(tokens)
        word_ids_map.extend([word_idx] * len(tokens))

    all_input_ids.append(tokenizer.sep_token_id)
    word_ids_map.append(-1)

    seq_len = len(all_input_ids)
    attention_mask = [1] * seq_len

    pad_len = max_len - seq_len
    all_input_ids += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len
    word_ids_map += [-1] * pad_len

    word_count = max(word_ids_map) + 1 if max(word_ids_map) >= 0 else 0
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

    word_tags = [O_TAG] * word_count
    spans = item.get('labels', [])
    sorted_spans = sorted(spans, key=lambda s: s[1] - s[0])

    for start_char, end_char, label_str in sorted_spans:
        b_tag = TAG2ID.get(f'B-{label_str}', O_TAG)
        i_tag = TAG2ID.get(f'I-{label_str}', O_TAG)
        if b_tag == O_TAG: continue
        
        first_token = True
        for w_idx in range(word_count):
            w_start, w_end = positions[w_idx]
            if w_start < end_char and w_end > start_char:
                if first_token:
                    word_tags[w_idx] = b_tag
                    first_token = False
                else:
                    word_tags[w_idx] = i_tag

    word_tags_padded = word_tags + [O_TAG] * (max_len - len(word_tags))

    return {
        'input_ids': all_input_ids,
        'attention_mask': attention_mask,
        'word_ids': word_ids_map,
        'word_tags': word_tags_padded,
        'word_count': word_count,
    }

def find_minority_tags(data_items):
    tag_counts = Counter()
    for item in data_items:
        for t in item['word_tags']:
            if t != O_TAG:
                tag_counts[t] += 1
    
    b_counts = {t: c for t, c in tag_counts.items() if BIO_TAGS[t].startswith('B-')}
    if not b_counts: return set()
    
    median = sorted(b_counts.values())[len(b_counts) // 2]
    threshold = median // 2
    minority = {t for t, c in b_counts.items() if c < threshold}
    return minority

class E2EDataset(Dataset):
    """
    Joint / E2E PhoBERT Dataset using proper VnCoreNLP segmentation 
    and word-level feature alignment for CRF.
    Includes data oversampling for minority tags.
    """
    def __init__(self, items, tokenizer_name="vinai/phobert-base", max_len=128, is_train=False):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        print(f"Tokenizing {len(items)} items...")
        processed_data = []
        for item in items:
            processed_data.append(tokenize_and_align(item, self.tokenizer, max_len))
            
        if is_train:
            minority_tags = find_minority_tags(processed_data)
            augmented_data = list(processed_data)
            added = 0
            for item in processed_data:
                if any(t in minority_tags for t in item['word_tags']):
                    augmented_data.append(item)
                    added += 1
            print(f"Oversampling added {added} minority samples.")
            self.data = augmented_data
        else:
            self.data = processed_data

        self.input_ids = torch.tensor([d['input_ids'] for d in self.data], dtype=torch.long)
        self.attention_masks = torch.tensor([d['attention_mask'] for d in self.data], dtype=torch.long)
        self.word_ids = torch.tensor([d['word_ids'] for d in self.data], dtype=torch.long)
        self.word_tags = torch.tensor([d['word_tags'] for d in self.data], dtype=torch.long)
        self.word_counts = torch.tensor([d['word_count'] for d in self.data], dtype=torch.long)

    def __len__(self): return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'word_ids': self.word_ids[idx],
            'word_tags': self.word_tags[idx],
            'word_count': self.word_counts[idx]
        }
