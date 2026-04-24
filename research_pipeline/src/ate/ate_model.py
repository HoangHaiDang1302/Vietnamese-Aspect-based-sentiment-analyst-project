"""
ATE Models - Aspect Term Extraction (Sequence Labeling + CRF)
Hỗ trợ: TextCNN, RNN, LSTM, GRU, BiLSTM, BiGRU
Kế thừa kiến trúc SequenceCRF và CNNCRF từ baseline_all_models_crf.ipynb
"""
import torch
import torch.nn as nn
from torchcrf import CRF


class ATESequenceCRF(nn.Module):
    """BiRNN/LSTM/GRU + CRF cho Sequence Labeling (kế thừa từ baseline)"""

    def __init__(self, vocab_size, emb_dim, hidden_dim, num_tags,
                 pretrained_emb=None, n_layers=2, dropout=0.3, pad_idx=0,
                 rnn_type='gru', bidir=True):
        super().__init__()
        self.bidir = bidir
        rnn_out = hidden_dim * 2 if bidir else hidden_dim

        # Embedding: dùng pre-trained nếu có
        if pretrained_emb is not None:
            self.emb = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_emb), freeze=False, padding_idx=pad_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        self.drop = nn.Dropout(dropout)

        # Chọn loại RNN
        if rnn_type == 'lstm':
            rnn_cls = nn.LSTM
        elif rnn_type == 'gru':
            rnn_cls = nn.GRU
        else:
            rnn_cls = nn.RNN

        self.rnn = rnn_cls(emb_dim, hidden_dim, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0, bidirectional=bidir)

        self.hidden2tag = nn.Sequential(
            nn.Linear(rnn_out, rnn_out // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_out // 2, num_tags)
        )
        self.crf = CRF(num_tags, batch_first=True)

    def _get_emissions(self, seqs, lens):
        emb = self.drop(self.emb(seqs))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        output, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, total_length=seqs.size(1))
        return self.hidden2tag(self.drop(output))

    def forward(self, seqs, mask=None, labels=None, lens=None):
        if lens is None:
            lens = mask.sum(dim=1)
        emissions = self._get_emissions(seqs, lens)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)


class ATECNNCRF(nn.Module):
    """TextCNN + CRF cho Sequence Labeling (kế thừa từ baseline)"""

    def __init__(self, vocab_size, emb_dim, hidden_dim, num_tags,
                 pretrained_emb=None, pad_idx=0, dropout=0.3):
        super().__init__()
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
        self.hidden2tag = nn.Sequential(
            nn.Linear(conv_out, conv_out // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(conv_out // 2, num_tags)
        )
        self.crf = CRF(num_tags, batch_first=True)

    def _get_emissions(self, seqs):
        emb = self.drop(self.emb(seqs)).transpose(1, 2)
        conv_outs = [torch.relu(conv(emb)) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1).transpose(1, 2)
        return self.hidden2tag(self.drop(out))

    def forward(self, seqs, mask=None, labels=None, lens=None):
        emissions = self._get_emissions(seqs)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)


def build_ate_model(model_type, vocab_size, emb_dim, hidden_dim, num_tags,
                    pretrained_emb=None, n_layers=2, dropout=0.3, pad_idx=0):
    """Factory function: tạo model ATE theo tên."""
    if model_type == 'TextCNN':
        return ATECNNCRF(vocab_size, emb_dim, hidden_dim, num_tags,
                         pretrained_emb, pad_idx, dropout)

    rnn_map = {
        'RNN': ('rnn', False),
        'LSTM': ('lstm', False),
        'GRU': ('gru', False),
        'BiLSTM': ('lstm', True),
        'BiGRU': ('gru', True),
    }
    rnn_type, bidir = rnn_map[model_type]
    return ATESequenceCRF(vocab_size, emb_dim, hidden_dim, num_tags,
                          pretrained_emb, n_layers, dropout, pad_idx, rnn_type, bidir)
