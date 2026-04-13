"""
Text preprocessing utilities for Vietnamese ABSA.
"""
import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to NFC form."""
    return unicodedata.normalize('NFC', text)


def clean_text(text: str) -> str:
    """Basic text cleaning for Vietnamese reviews."""
    text = normalize_unicode(text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def simple_word_tokenize(text: str) -> list:
    """Simple whitespace tokenizer (no word segmentation dependency)."""
    return text.lower().split()
