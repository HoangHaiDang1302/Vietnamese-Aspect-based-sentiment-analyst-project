"""
ASC Models - Aspect Sentiment Classification
Hỗ trợ: TextCNN, RNN, LSTM, GRU, BiLSTM, BiGRU
Phân loại cảm xúc cho một khía cạnh cho trước (3 classes: POS/NEG/NEU)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASCSequenceModel(nn.Module):
    """RNN/LSTM/GRU Classifier cho Sentiment"""

    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes=3,
                 pretrained_emb=None, n_layers=2, dropout=0.3, pad_idx=0,
                 rnn_type='gru', bidir=True):
        super().__init__()
        self.bidir = bidir
        self.rnn_type_name = rnn_type

        if pretrained_emb is not None:
            self.emb = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_emb), freeze=False, padding_idx=pad_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        self.drop = nn.Dropout(dropout)

        if rnn_type == 'lstm':
            rnn_cls = nn.LSTM
        elif rnn_type == 'gru':
            rnn_cls = nn.GRU
        else:
            rnn_cls = nn.RNN

        self.rnn = rnn_cls(emb_dim, hidden_dim, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0, bidirectional=bidir)

        out_dim = hidden_dim * 2 if bidir else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes)
        )

    def forward(self, x, lens=None):
        emb = self.drop(self.emb(x))

        if lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
            _, hidden = self.rnn(packed)
        else:
            _, hidden = self.rnn(emb)

        if self.rnn_type_name == 'lstm':
            hidden = hidden[0]  # h_n only

        if self.bidir:
            feats = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            feats = hidden[-1]

        return self.fc(feats)


class ASCCNNModel(nn.Module):
    """TextCNN Classifier cho Sentiment"""

    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes=3,
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
        self.fc = nn.Sequential(
            nn.Linear(conv_out, conv_out // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(conv_out // 2, num_classes)
        )

    def forward(self, x, lens=None):
        emb = self.drop(self.emb(x)).transpose(1, 2)
        conv_outs = [F.relu(conv(emb)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_outs]
        feats = torch.cat(pooled, dim=1)
        return self.fc(feats)


def build_asc_model(model_type, vocab_size, emb_dim, hidden_dim, num_classes=3,
                    pretrained_emb=None, n_layers=2, dropout=0.3, pad_idx=0):
    """Factory function: tạo model ASC theo tên."""
    if model_type == 'TextCNN':
        return ASCCNNModel(vocab_size, emb_dim, hidden_dim, num_classes,
                           pretrained_emb, pad_idx, dropout)

    rnn_map = {
        'RNN': ('rnn', False),
        'LSTM': ('lstm', False),
        'GRU': ('gru', False),
        'BiLSTM': ('lstm', True),
        'BiGRU': ('gru', True),
    }
    rnn_type, bidir = rnn_map[model_type]
    return ASCSequenceModel(vocab_size, emb_dim, hidden_dim, num_classes,
                            pretrained_emb, n_layers, dropout, pad_idx, rnn_type, bidir)
