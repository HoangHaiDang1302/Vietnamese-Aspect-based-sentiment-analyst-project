"""
Predictor Service — loads models and runs inference.
Initializes once at startup, shared across requests.
"""
import os
import json
import logging
import numpy as np
import torch
from typing import Dict, List, Optional

from ..config import (
    BIGRU_CONFIG, PHOBERT_CONFIG, DATA_DIR,
    BIO_TAGS, NUM_TAGS, ASPECTS, SENTIMENTS, LABEL_NAMES,
)
from ..models.bigru_crf import BiGRUCRF
from ..utils.bio_utils import bio_tags_to_spans
from ..utils.text_utils import clean_text

logger = logging.getLogger(__name__)

# Word2Vec vocabulary constants
PAD_IDX = 0
UNK_IDX = 1


class ABSAPredictor:
    """
    Manages model loading and inference for ABSA prediction.
    Supports BiGRU-CRF (default) and PhoBERT-CRF.
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models: Dict[str, torch.nn.Module] = {}
        self.word2idx: Optional[Dict[str, int]] = None
        self._loaded = False

    @property
    def available_models(self) -> List[str]:
        return list(self.models.keys())

    def load_bigru(self) -> bool:
        """Load BiGRU-CRF model with Word2Vec vocab from training data."""
        model_path = BIGRU_CONFIG["model_path"]
        if not os.path.exists(model_path):
            logger.warning(f"BiGRU-CRF weights not found at {model_path}")
            return False

        try:
            logger.info("Building Word2Vec vocabulary from data...")
            self._build_vocabulary()

            vocab_size = len(self.word2idx)
            emb_dim = BIGRU_CONFIG["w2v_dim"]
            hidden_dim = BIGRU_CONFIG["hidden_dim"]

            # Build embedding matrix (random init, weights loaded from checkpoint)
            emb_matrix = np.random.normal(0, 0.1, (vocab_size, emb_dim)).astype(np.float32)
            emb_matrix[PAD_IDX] = 0

            model = BiGRUCRF(
                vocab_size=vocab_size,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                num_tags=NUM_TAGS,
                pretrained_emb=emb_matrix,
                n_layers=BIGRU_CONFIG["num_layers"],
                dropout=BIGRU_CONFIG["dropout"],
                pad_idx=PAD_IDX,
            ).to(self.device)

            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            self.models["bigru_crf"] = model
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"✅ BiGRU-CRF loaded ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load BiGRU-CRF: {e}")
            return False

    def _build_vocabulary(self):
        """Build word2idx from training data (same as training pipeline)."""
        from gensim.models import Word2Vec

        texts = []
        for split in ['train.jsonl', 'dev.jsonl', 'test.jsonl']:
            path = DATA_DIR / split
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        texts.append(json.loads(line.strip())['text'])

        all_sentences = [t.lower().split() for t in texts]
        w2v = Word2Vec(
            all_sentences, vector_size=BIGRU_CONFIG["w2v_dim"],
            window=5, min_count=2, workers=4, epochs=20, sg=1, seed=42,
        )

        self.word2idx = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
        for i, w in enumerate(w2v.wv.index_to_key):
            self.word2idx[w] = i + 2

    def predict(self, text: str, model_name: str = "bigru_crf") -> List[dict]:
        """
        Run prediction on input text.
        
        Args:
            text: Input Vietnamese review text
            model_name: Which model to use
            
        Returns:
            List of aspect-sentiment span dicts
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded. Available: {self.available_models}")

        text = clean_text(text)

        if model_name == "bigru_crf":
            return self._predict_bigru(text)
        else:
            raise ValueError(f"Model '{model_name}' inference not implemented yet")

    def _predict_bigru(self, text: str) -> List[dict]:
        """Run BiGRU-CRF inference."""
        model = self.models["bigru_crf"]
        max_len = BIGRU_CONFIG["max_len"]

        words = text.lower().split()[:max_len]
        seq = [self.word2idx.get(w, UNK_IDX) for w in words]
        seq_len = len(seq)

        if seq_len == 0:
            return []

        tensor_seq = torch.tensor([seq]).to(self.device)
        tensor_len = torch.tensor([seq_len])
        tensor_mask = torch.ones(1, seq_len, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            preds = model(tensor_seq, tensor_len, tensor_mask)
            tags = preds['tags'][0][:seq_len]
            spans = bio_tags_to_spans(tags, seq_len)

        # Map word indices back to character positions in original text
        pos = 0
        text_lower = text.lower()
        char_positions = []
        for w in words:
            idx = text_lower.find(w, pos)
            if idx == -1:
                idx = pos
            char_positions.append((idx, idx + len(w)))
            pos = idx + len(w)

        results = []
        for label_str, s, e_exc in spans:
            if "#" not in label_str:
                continue
            aspect, sentiment = label_str.split('#')
            start_char = char_positions[s][0]
            end_char = char_positions[e_exc - 1][1]

            results.append({
                "aspect": aspect,
                "sentiment": sentiment,
                "text": text[start_char:end_char].strip(),
                "start": start_char,
                "end": end_char,
            })

        return results


# Singleton instance
predictor = ABSAPredictor()
