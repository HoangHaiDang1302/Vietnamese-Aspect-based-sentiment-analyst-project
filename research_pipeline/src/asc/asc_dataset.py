import torch
from torch.utils.data import Dataset
from ..utils.preprocess import tokenize_baseline

class ASCDataset(Dataset):
    """
    ASC: Phân loại Cảm Xúc. 
    Kỹ thuật được bổ sung: Nhúng thêm vào chuỗi text thẻ [ASP] bao quanh khía cạnh để RNN nhận diện.
    Đầu vào: Câu Text + Đánh dấu vị trí.
    Đầu ra: POSITIVE (0), NEGATIVE (1), NEUTRAL (2)
    """
    def __init__(self, items, word2idx, max_len=128):
        self.seqs, self.lens, self.labels = [], [], []
        
        d_map = {"POSITIVE": 0, "NEGATIVE": 1, "NEUTRAL": 2}
        
        for item in items:
            orig_text = item['text']  # Đã được segment (có _ nối từ ghép)
            spans = item.get('labels', [])
            
            # Nếu 1 câu có nhiều aspect, tách thành NHIỀU SAMPLES (Mỗi Aspect một dòng riêng biệt)
            for s_char, e_char, raw_label in spans:
                if "#" in raw_label:
                    aspect, sentiment = raw_label.split("#")
                    if sentiment in d_map:
                        # Tìm vị trí word-level tương ứng với char offset trên text gốc
                        # Chèn [ASP] marker vào segmented text
                        words = orig_text.split()
                        pos = 0
                        start_word, end_word = None, None
                        for w_idx, w in enumerate(words):
                            w_clean = w.replace('_', ' ')
                            w_len = len(w_clean)
                            if start_word is None and pos + w_len > s_char:
                                start_word = w_idx
                            if pos < e_char:
                                end_word = w_idx + 1
                            pos += w_len + 1  # +1 for space
                        
                        if start_word is not None and end_word is not None:
                            # Insert [ASP] markers at word boundaries
                            marked_words = (words[:start_word] + ['[ASP]'] + 
                                          words[start_word:end_word] + ['[ASP]'] + 
                                          words[end_word:])
                            marked_text = ' '.join(marked_words)
                        else:
                            marked_text = orig_text
                        
                        seq, length = tokenize_baseline(marked_text, word2idx, max_len)
                        
                        self.seqs.append(seq)
                        self.lens.append(length)
                        self.labels.append(d_map[sentiment])
                        
        self.seqs = torch.tensor(self.seqs, dtype=torch.long)
        self.lens = torch.tensor(self.lens, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return {'seq': self.seqs[i], 'len': self.lens[i], 'label': self.labels[i]}
