"""
E2E Model - End-to-End ABSA with PhoBERT + CRF
Học chung hệ nhãn Unified Label (Ví dụ: B-CAMERA#POSITIVE)
Fix: Tính toán word-level emissions dựa trên subword đầu tiên của mỗi từ (word_ids).
"""
import torch
import torch.nn as nn
from transformers import RobertaModel
from torchcrf import CRF


class E2EPhoBertCRF(nn.Module):
    """
    PhoBERT + CRF cho End-to-End ABSA.
    Chỉ trích xuất token đầu tiên của mỗi word (theo word_ids) để đưa qua CRF.
    """
    def __init__(self, num_unified_tags, dropout=0.3):
        super().__init__()
        self.phobert = RobertaModel.from_pretrained("vinai/phobert-base")

        self.hidden_size = self.phobert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(self.hidden_size, num_unified_tags)
        self.crf = CRF(num_unified_tags, batch_first=True)
        self.num_tags = num_unified_tags

    def _get_word_emissions(self, input_ids, attention_mask, word_ids, word_counts):
        """
        Run PhoBERT, then extract FIRST subword representation per word.
        Returns: emissions (batch, max_words, num_tags), word_mask (batch, max_words)
        """
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)  # (B, seq_len, 768)

        batch_size = input_ids.size(0)
        max_words = word_counts.max().item()
        
        if max_words == 0:
            max_words = 1  # prevent error

        word_emissions = torch.zeros(batch_size, max_words, self.hidden_size,
                                     device=input_ids.device)
        word_mask = torch.zeros(batch_size, max_words, dtype=torch.bool,
                                device=input_ids.device)

        for b in range(batch_size):
            wc = word_counts[b].item()
            if wc > 0:
                word_mask[b, :wc] = True
            
            seen_words = set()
            for pos in range(word_ids.size(1)):
                wid = word_ids[b, pos].item()
                if wid >= 0 and wid < wc and wid not in seen_words:
                    word_emissions[b, wid] = sequence_output[b, pos]
                    seen_words.add(wid)

        emissions = self.hidden2tag(word_emissions)  # (B, max_words, num_tags)
        return emissions, word_mask

    def forward(self, input_ids, attention_mask, word_ids, word_counts, labels=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            word_ids: (batch, seq_len) mapping subwords to words
            word_counts: (batch,) number of words per sequence
            labels: (batch, seq_len) word_tags padded to max_len
        """
        emissions, word_mask = self._get_word_emissions(
            input_ids, attention_mask, word_ids, word_counts)

        if labels is not None:
            max_words = word_counts.max().item()
            if max_words == 0:
                max_words = 1
            # Lấy tags tương ứng với max_words
            tags = labels[:, :max_words]
            loss = -self.crf(emissions, tags, mask=word_mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=word_mask)
