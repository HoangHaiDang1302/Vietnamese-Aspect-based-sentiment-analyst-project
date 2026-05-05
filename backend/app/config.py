"""
Application Configuration
"""
import os
from pathlib import Path

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# === Model Config ===
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "bigru_crf")  # "bigru_crf" or "phobert_crf"

# BiGRU-CRF Config
BIGRU_CONFIG = {
    "model_path": MODELS_DIR / "best_e2e_baseline_BiGRU-CRF.pt",
    "w2v_path": MODELS_DIR / "word2vec.model",
    "w2v_dim": 150,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "max_len": 128,
}

# PhoBERT-CRF Config
PHOBERT_CONFIG = {
    "model_path": MODELS_DIR / "best_e2e_phobert.pt",
    "model_name": "vinai/phobert-base-v2",
    "max_len": 256,
    "dropout": 0.1,
}

# === ABSA Labels ===
ASPECTS = [
    "CAMERA", "FEATURES", "PERFORMANCE", "DESIGN", "PRICE",
    "GENERAL", "SCREEN", "BATTERY", "STORAGE", "SER&ACC"
]
SENTIMENTS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
LABEL_NAMES = [f"{a}#{s}" for a in ASPECTS for s in SENTIMENTS]
NUM_LABELS = len(LABEL_NAMES)  # 30

# BIO Tags
BIO_TAGS = ['O']
for label in LABEL_NAMES:
    BIO_TAGS.append(f'B-{label}')
    BIO_TAGS.append(f'I-{label}')
NUM_TAGS = len(BIO_TAGS)  # 61

LABEL2ID = {n: i for i, n in enumerate(LABEL_NAMES)}
TAG2ID = {t: i for i, t in enumerate(BIO_TAGS)}

# === Server Config ===
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
