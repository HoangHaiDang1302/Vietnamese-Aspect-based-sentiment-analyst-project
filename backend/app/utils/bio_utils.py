"""
BIO tag processing utilities.
Convert between BIO tag sequences and structured spans.
"""
from typing import List, Tuple
from ..config import BIO_TAGS, NUM_TAGS


def bio_tags_to_spans(tag_ids: List[int], max_tokens: int) -> List[Tuple[str, int, int]]:
    """
    Convert BIO tag ID sequence to structured spans.
    
    Args:
        tag_ids: List of tag IDs
        max_tokens: Maximum number of tokens to process
        
    Returns:
        List of (label_name, start_idx, end_idx) tuples
    """
    spans = []
    current_label = None
    current_start = None

    for t in range(min(len(tag_ids), max_tokens)):
        tag_id = tag_ids[t]
        tag_name = BIO_TAGS[tag_id] if tag_id < NUM_TAGS else 'O'

        if tag_name.startswith('B-'):
            if current_label is not None:
                spans.append((current_label, current_start, t))
            current_label = tag_name[2:]
            current_start = t

        elif tag_name.startswith('I-'):
            label = tag_name[2:]
            if current_label != label:
                if current_label is not None:
                    spans.append((current_label, current_start, t))
                current_label = label
                current_start = t
        else:
            if current_label is not None:
                spans.append((current_label, current_start, t))
                current_label = None

    if current_label is not None:
        spans.append((current_label, current_start, len(tag_ids)))

    return spans
