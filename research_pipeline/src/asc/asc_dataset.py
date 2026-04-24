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
            orig_text = item['text']
            spans = item.get('labels', [])
            
            # Nếu 1 câu có nhiều aspect, tách thành NHIỀU SAMPLES (Mỗi Aspect một dòng riêng biệt)
            for s_char, e_char, raw_label in spans:
                if "#" in raw_label:
                    aspect, sentiment = raw_label.split("#")
                    if sentiment in d_map:
                        # Kỹ thuật bọc ngụy trang để làm nổi bật vị trí Aspect
                        marked_text = orig_text[:s_char] + f" [ASP] {orig_text[s_char:e_char]} [ASP] " + orig_text[e_char:]
                        
                        seq, length = tokenize_baseline(marked_text, word2idx, max_len)
                        
                        self.seqs.append(seq)
                        self.lens.append(length)
                        self.labels.append(d_map[sentiment])
                        
        self.seqs = torch.tensor(self.seqs, dtype=torch.long)
        self.lens = torch.tensor(self.lens, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return {'seq': self.seqs[i], 'len': self.lens[i], 'label': self.labels[i]}
