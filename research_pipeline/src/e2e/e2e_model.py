"""
E2E Model - End-to-End ABSA with PhoBERT + CRF
Học chung hệ nhãn Unified Label (Ví dụ: B-CAMERA#POSITIVE)
Fix: Xử lý special tokens (-100) đúng cách cho CRF.
"""
import torch
import torch.nn as nn
from transformers import RobertaModel
from torchcrf import CRF


class E2EPhoBertCRF(nn.Module):
    """
    PhoBERT + CRF cho End-to-End ABSA.
    Xử lý special tokens: CRF chỉ nhận emissions tại các vị trí word thực sự,
    bỏ qua hoàn toàn <s>, </s>, <pad> và subword continuation tokens.
    """
    def __init__(self, num_unified_tags, dropout=0.3):
        super().__init__()
        self.phobert = RobertaModel.from_pretrained("vinai/phobert-base")

        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(self.phobert.config.hidden_size, num_unified_tags)
        self.crf = CRF(num_unified_tags, batch_first=True)
        self.num_tags = num_unified_tags

    def forward(self, input_ids, attention_mask, labels=None, word_mask=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) - 1 cho token thực, 0 cho pad
            labels: (batch, seq_len) - BIO tag IDs, -100 cho special tokens/subwords
            word_mask: (batch, seq_len) - boolean, True chỉ tại vị trí first-subword
                       Nếu không truyền, sẽ tự tính từ labels != -100
        """
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.hidden2tag(sequence_output)  # (batch, seq_len, num_tags)

        # Cung cấp mask hoàn hảo cho CRF (Chỉ bao gồm phần tử gốc, bỏ phần Pad ở đuôi).
        # Sự liên tiếp là đặc biệt sống còn đối với thư viện pytorch-crf.
        crf_mask = attention_mask.bool()

        if labels is not None:
            # -100 chỉ tồn tại ở các khoảng Padding do dataset truyền
            clean_labels = labels.clone()
            clean_labels[clean_labels == -100] = 0

            loss = -self.crf(emissions, tags=clean_labels, mask=crf_mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=crf_mask)
