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
    DEFAULT_MODEL,
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
    Supports PhoBERT-CRF (default) and BiGRU-CRF.
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
        """Load BiGRU-CRF model with Word2Vec vocab from pretrained model."""
        model_path = BIGRU_CONFIG["model_path"]
        if not os.path.exists(model_path):
            logger.warning(f"BiGRU-CRF weights not found at {model_path}")
            return False

        try:
            # 1. Load state_dict first to check the expected vocabulary size
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            checkpoint_vocab_size = state_dict['emb.weight'].shape[0]
            logger.info(f"BiGRU-CRF checkpoint expects vocab_size={checkpoint_vocab_size}")

            # 2. Load vocabulary with the matching limit
            self._load_vocabulary(limit=checkpoint_vocab_size - 2)
            
            vocab_size = len(self.word2idx)
            if vocab_size != checkpoint_vocab_size:
                 logger.warning(f"Vocab mismatch: loaded {vocab_size}, but checkpoint needs {checkpoint_vocab_size}. Adjusting...")
                 # This shouldn't happen with the limit, but just in case

            emb_dim = BIGRU_CONFIG["w2v_dim"]
            hidden_dim = BIGRU_CONFIG["hidden_dim"]

            # Build embedding matrix
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

            model.load_state_dict(state_dict)
            model.eval()

            self.models["bigru_crf"] = model
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"✅ BiGRU-CRF loaded ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load BiGRU-CRF: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def load_phobert(self) -> bool:
        """Load PhoBERT-CRF model."""
        from ..models.phobert_crf import PhoBERTCRF
        from transformers import AutoTokenizer

        model_path = PHOBERT_CONFIG["model_path"]
        if not os.path.exists(model_path):
            logger.warning(f"PhoBERT-CRF weights not found at {model_path}")
            return False

        try:
            logger.info("Loading PhoBERT tokenizer & model (this may take a few seconds)...")
            self.tokenizer = AutoTokenizer.from_pretrained(PHOBERT_CONFIG["model_name"])

            model = PhoBERTCRF(
                model_name=PHOBERT_CONFIG["model_name"],
                num_tags=NUM_TAGS,
                dropout=PHOBERT_CONFIG["dropout"],
                pretrained=False
            ).to(self.device)

            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            self.models["phobert_crf"] = model
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"✅ PhoBERT-CRF loaded ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load PhoBERT-CRF: {e}")
            return False

    def _load_vocabulary(self, limit: Optional[int] = None):
        """Load word2idx from pre-trained Word2Vec model."""
        from gensim.models import Word2Vec
        w2v_path = BIGRU_CONFIG.get("w2v_path")
        if not w2v_path or not os.path.exists(w2v_path):
            raise FileNotFoundError(f"Word2Vec model not found at {w2v_path}")

        w2v = Word2Vec.load(str(w2v_path))
        self.word2idx = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
        
        keys = w2v.wv.index_to_key
        if limit is not None:
            keys = keys[:limit]
            
        for i, w in enumerate(keys):
            self.word2idx[w] = i + 2

    def predict(self, text: str, model_name: str = DEFAULT_MODEL) -> List[dict]:
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
        elif model_name == "phobert_crf":
            return self._predict_phobert(text)
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

    def _predict_phobert(self, text: str) -> List[dict]:
        """Run PhoBERT-CRF inference."""
        model = self.models["phobert_crf"]
        max_len = PHOBERT_CONFIG["max_len"]

        from ..utils.text_utils import simple_word_tokenize
        words = simple_word_tokenize(text)
        if not words:
            return []

        # Tokenize subwords and align
        input_ids = [self.tokenizer.cls_token_id]
        word_ids_list = [-1]
        
        for idx, word in enumerate(words):
            subwords = self.tokenizer.encode(word, add_special_tokens=False)
            if not subwords:
                subwords = [self.tokenizer.unk_token_id]
            
            # Stop if we exceed max_len (minus 1 for SEP)
            if len(input_ids) + len(subwords) >= max_len:
                break
                
            input_ids.extend(subwords)
            word_ids_list.extend([idx] * len(subwords))
            
        input_ids.append(self.tokenizer.sep_token_id)
        word_ids_list.append(-1)
        valid_words_len = len(set([x for x in word_ids_list if x != -1]))

        tensor_ids = torch.tensor([input_ids]).to(self.device)
        tensor_mask = torch.ones_like(tensor_ids).to(self.device)
        tensor_word_ids = torch.tensor([word_ids_list]).to(self.device)
        tensor_word_count = torch.tensor([valid_words_len]).to(self.device)

        with torch.no_grad():
            preds = model(tensor_ids, tensor_mask, tensor_word_ids, tensor_word_count)
            tags = preds['tags'][0][:valid_words_len]
            spans = bio_tags_to_spans(tags, valid_words_len)

        # Reconstruct characters
        pos = 0
        char_positions = []
        text_lower = text.lower()
        for w in words[:valid_words_len]:
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
