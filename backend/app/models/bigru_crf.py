"""
BiGRU-CRF Model for BIO Sequence Labeling
Lightweight model (~12MB) suitable for CPU deployment.
"""
import torch
import torch.nn as nn
from torchcrf import CRF


class BiGRUCRF(nn.Module):
    """BiGRU + CRF for Aspect-Based Sentiment Analysis via BIO tagging."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 150,
        hidden_dim: int = 256,
        num_tags: int = 61,
        pretrained_emb=None,
        n_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.bidir = True
        rnn_out = hidden_dim * 2

        if pretrained_emb is not None:
            self.emb = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_emb), freeze=False, padding_idx=pad_idx
            )
        else:
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            emb_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )
        self.hidden2tag = nn.Sequential(
            nn.Linear(rnn_out, rnn_out // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_out // 2, num_tags),
        )
        self.crf = CRF(num_tags, batch_first=True)

    def _get_emissions(self, seqs, lens):
        emb = self.drop(self.emb(seqs))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        output, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, total_length=seqs.size(1)
        )
        return self.hidden2tag(self.drop(output))

    def forward(self, seqs, lens, mask, tags=None):
        emissions = self._get_emissions(seqs, lens)
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return {'loss': loss}
        else:
            best_tags = self.crf.decode(emissions, mask=mask)
            return {'tags': best_tags}
