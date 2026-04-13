"""
Text Processing Utilities for Vietnamese ABSA

This module provides text preprocessing and augmentation functions
for Vietnamese Aspect-Based Sentiment Analysis.
"""

import re
import random
import unicodedata
from typing import List, Optional

# Try to import underthesea for Vietnamese word segmentation
try:
    from underthesea import word_tokenize
    SEGMENT_AVAILABLE = True
except ImportError:
    SEGMENT_AVAILABLE = False
    print("⚠️ underthesea not available. Word segmentation disabled.")


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text to NFC form.
    This ensures consistent representation of Vietnamese characters.
    """
    return unicodedata.normalize('NFC', text)


def clean_text(text: str) -> str:
    """
    Clean and normalize Vietnamese text.
    
    Processing steps:
    1. Normalize unicode (NFC)
    2. Lowercase
    3. Remove URLs
    4. Remove email addresses
    5. Remove phone numbers (VN format)
    6. Normalize repeated characters
    7. Remove emojis (preserve Vietnamese chars)
    8. Normalize whitespace
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Normalize unicode
    text = normalize_unicode(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers (VN format: 10-11 digits starting with 0)
    text = re.sub(r'\b0\d{9,10}\b', '', text)
    
    # Normalize repeated characters (e.g., "quáaaa" -> "quáa")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Remove emojis but keep Vietnamese characters
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002500-\U00002BEF"  # chinese/japanese/korean chars
        u"\U00002702-\U000027B0"  # dingbats
        u"\U0001F900-\U0001F9FF"  # supplemental symbols
        u"\U0001FA00-\U0001FA6F"  # chess symbols
        u"\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
        u"\U00002600-\U000026FF"  # misc symbols
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def segment_vietnamese(text: str) -> str:
    """
    Apply Vietnamese word segmentation using underthesea.
    
    Args:
        text: Input text (should be cleaned first)
        
    Returns:
        Word-segmented text with underscore-connected compound words
        Returns original text if segmentation fails or unavailable
    """
    if not SEGMENT_AVAILABLE:
        return text
    
    try:
        return word_tokenize(text, format='text')
    except Exception:
        return text


def preprocess_text(text: str, 
                    clean: bool = True, 
                    segment: bool = True) -> str:
    """
    Full text preprocessing pipeline.
    
    Args:
        text: Input text
        clean: Whether to apply cleaning
        segment: Whether to apply word segmentation
        
    Returns:
        Preprocessed text
    """
    if clean:
        text = clean_text(text)
    if segment and SEGMENT_AVAILABLE:
        text = segment_vietnamese(text)
    return text


# ============================================================
# TEXT AUGMENTATION
# ============================================================

class TextAugmenter:
    """
    Simple text augmentation for Vietnamese text.
    
    Supports:
    - Random word swap
    - Random word deletion
    - Random combination of above
    """
    
    def __init__(self, 
                 swap_prob: float = 0.1, 
                 delete_prob: float = 0.1,
                 seed: Optional[int] = None):
        """
        Initialize augmenter.
        
        Args:
            swap_prob: Probability of swapping each word pair
            delete_prob: Probability of deleting each word
            seed: Random seed for reproducibility
        """
        self.swap_prob = swap_prob
        self.delete_prob = delete_prob
        if seed is not None:
            random.seed(seed)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """
        Randomly swap n pairs of word positions.
        
        Args:
            text: Input text
            n: Number of swaps to perform
            
        Returns:
            Text with swapped words
        """
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_delete(self, text: str) -> str:
        """
        Randomly delete words with probability delete_prob.
        
        Args:
            text: Input text
            
        Returns:
            Text with some words deleted
        """
        words = text.split()
        if len(words) <= 2:
            return text
        
        new_words = [w for w in words if random.random() > self.delete_prob]
        
        # Ensure at least 2 words remain
        if len(new_words) < 2:
            # Keep first and last word at minimum
            return f"{words[0]} {words[-1]}"
        
        return ' '.join(new_words)
    
    def random_insert_duplicate(self, text: str) -> str:
        """
        Randomly duplicate and insert a word at random position.
        
        Args:
            text: Input text
            
        Returns:
            Text with duplicated word inserted
        """
        words = text.split()
        if len(words) < 1:
            return text
        
        word_to_dup = random.choice(words)
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, word_to_dup)
        
        return ' '.join(words)
    
    def augment(self, text: str, method: Optional[str] = None) -> str:
        """
        Apply random augmentation.
        
        Args:
            text: Input text
            method: Specific method ('swap', 'delete', 'duplicate')
                   If None, randomly chooses one
            
        Returns:
            Augmented text
        """
        if method is None:
            method = random.choice(['swap', 'delete', 'duplicate'])
        
        if method == 'swap':
            return self.random_swap(text)
        elif method == 'delete':
            return self.random_delete(text)
        elif method == 'duplicate':
            return self.random_insert_duplicate(text)
        else:
            return text
    
    def augment_multiple(self, text: str, n: int = 3) -> List[str]:
        """
        Generate multiple augmented versions.
        
        Args:
            text: Input text
            n: Number of augmented versions to generate
            
        Returns:
            List of augmented texts
        """
        augmented = []
        methods = ['swap', 'delete', 'duplicate']
        
        for i in range(n):
            method = methods[i % len(methods)]
            aug_text = self.augment(text, method=method)
            augmented.append(aug_text)
        
        return augmented


# ============================================================
# BATCH PROCESSING
# ============================================================

def batch_preprocess(texts: List[str], 
                     clean: bool = True, 
                     segment: bool = True,
                     show_progress: bool = True) -> List[str]:
    """
    Preprocess a batch of texts.
    
    Args:
        texts: List of input texts
        clean: Whether to apply cleaning
        segment: Whether to apply word segmentation
        show_progress: Whether to show progress bar
        
    Returns:
        List of preprocessed texts
    """
    processed = []
    
    iterator = texts
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Preprocessing")
        except ImportError:
            pass
    
    for text in iterator:
        processed.append(preprocess_text(text, clean=clean, segment=segment))
    
    return processed


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    # Test cleaning
    test_text = "Điện thoại này quá tuyệt vờiiii!!! Pin trâu lắm 👍👍👍 mua ở tgdđ ok nhé"
    print(f"Original:  {test_text}")
    print(f"Cleaned:   {clean_text(test_text)}")
    print(f"Segmented: {preprocess_text(test_text)}")
    
    # Test augmentation
    augmenter = TextAugmenter(seed=42)
    clean = clean_text(test_text)
    print(f"\nAugmentation test:")
    print(f"  Swap:      {augmenter.augment(clean, 'swap')}")
    print(f"  Delete:    {augmenter.augment(clean, 'delete')}")
    print(f"  Duplicate: {augmenter.augment(clean, 'duplicate')}")
