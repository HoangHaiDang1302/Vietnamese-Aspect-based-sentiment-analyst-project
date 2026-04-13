"""
PhoBERT-CRF Model for BIO Sequence Labeling
High-accuracy model (~540MB) using pre-trained PhoBERT.
"""
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel


class PhoBERTCRF(nn.Module):
    """PhoBERT + Linear + CRF for ABSA via BIO Sequence Labeling."""

    def __init__(self, model_name: str, num_tags: int = 61, dropout: float = 0.1):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.phobert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def _get_word_emissions(self, input_ids, attention_mask, word_ids, word_count):
        """
        Run PhoBERT, then extract FIRST subword representation per word.
        Returns: emissions (batch, max_words, num_tags), word_mask (batch, max_words)
        """
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)

        batch_size = input_ids.size(0)
        max_words = word_count.max().item()

        word_emissions = torch.zeros(
            batch_size, max_words, self.hidden_size, device=input_ids.device
        )
        word_mask = torch.zeros(
            batch_size, max_words, dtype=torch.bool, device=input_ids.device
        )

        for b in range(batch_size):
            wc = word_count[b].item()
            word_mask[b, :wc] = True
            seen_words = set()
            for pos in range(word_ids.size(1)):
                wid = word_ids[b, pos].item()
                if wid >= 0 and wid < wc and wid not in seen_words:
                    word_emissions[b, wid] = sequence_output[b, pos]
                    seen_words.add(wid)

        emissions = self.classifier(word_emissions)
        return emissions, word_mask

    def forward(self, input_ids, attention_mask, word_ids, word_count, word_tags=None):
        emissions, word_mask = self._get_word_emissions(
            input_ids, attention_mask, word_ids, word_count
        )

        if word_tags is not None:
            max_words = word_count.max().item()
            tags = word_tags[:, :max_words]
            loss = -self.crf(emissions, tags, mask=word_mask, reduction='mean')
            return {'loss': loss}
        else:
            best_tags = self.crf.decode(emissions, mask=word_mask)
            return {'tags': best_tags}
