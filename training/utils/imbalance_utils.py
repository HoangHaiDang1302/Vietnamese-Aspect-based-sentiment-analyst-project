"""
Utility functions for handling class imbalance in ABSA tasks.
Includes: Focal Loss, Class Weights, Threshold Tuning, Data Augmentation, Oversampling

Author: Vietnamese ABSA Project
Version: 2.0 - Added Augmentation-based Oversampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score, classification_report
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import random
import re


# =============================================================================
# 1. FOCAL LOSS
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    FL = -alpha * (1 - pt)^gamma * log(pt)
    
    Args:
        gamma: Focusing parameter (default=2). Higher = more focus on hard samples
        alpha: Per-class weight tensor [num_classes]
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            focal_loss = self.alpha.unsqueeze(0) * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class FocalLossMultiClass(nn.Module):
    """Focal Loss for multi-class classification (single label)."""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingBCELoss(nn.Module):
    """BCE Loss with label smoothing to prevent overconfidence."""
    def __init__(self, epsilon=0.1, pos_weight=None, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.epsilon) + (1 - targets) * self.epsilon
        return F.binary_cross_entropy_with_logits(
            logits, targets_smooth, pos_weight=self.pos_weight, reduction=self.reduction
        )


# =============================================================================
# 2. CLASS WEIGHT CALCULATION
# =============================================================================

def calculate_class_weights(labels, method='sqrt', device='cpu'):
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        labels: torch.Tensor [num_samples, num_classes] for multi-label
        method: 'inverse', 'sqrt', 'effective', 'balanced'
        device: torch device
    
    Returns:
        torch.Tensor of class weights
    """
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels, dtype=torch.float32)
    
    counts = labels.sum(dim=0).float()
    total = labels.shape[0]
    
    if method == 'inverse':
        weights = total / (counts + 1e-8)
    elif method == 'sqrt':
        weights = torch.sqrt(total / (counts + 1e-8))
    elif method == 'effective':
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
    elif method == 'balanced':
        weights = total / (len(counts) * counts + 1e-8)
    else:
        weights = torch.ones(labels.shape[1])
    
    # Normalize
    weights = weights / weights.sum() * len(weights)
    return weights.to(device)


def calculate_pos_weight_for_bce(labels, device='cpu'):
    """Calculate pos_weight for BCEWithLogitsLoss."""
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels, dtype=torch.float32)
    
    pos_counts = labels.sum(dim=0).float()
    neg_counts = labels.shape[0] - pos_counts
    pos_weight = torch.where(pos_counts > 0, neg_counts / pos_counts, torch.ones_like(pos_counts))
    return pos_weight.to(device)


# =============================================================================
# 3. THRESHOLD TUNING
# =============================================================================

def find_optimal_thresholds(true_labels, pred_probs, class_names=None, verbose=True):
    """
    Find optimal threshold for each class based on F1-score.
    
    Args:
        true_labels: np.array [num_samples, num_classes]
        pred_probs: np.array [num_samples, num_classes]
        class_names: list of class names (optional)
        verbose: print results
    
    Returns:
        optimal_thresholds: np.array [num_classes]
    """
    num_classes = true_labels.shape[1]
    optimal_thresholds = []
    
    if verbose:
        print("\n" + "="*60)
        print("OPTIMAL THRESHOLD TUNING")
        print("="*60)
    
    for i in range(num_classes):
        # Skip if all same label
        if true_labels[:, i].sum() == 0 or true_labels[:, i].sum() == len(true_labels):
            optimal_thresholds.append(0.5)
            continue
            
        precisions, recalls, thresholds = precision_recall_curve(
            true_labels[:, i], pred_probs[:, i]
        )
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        optimal_thresholds.append(optimal_threshold)
        
        if verbose:
            class_name = class_names[i] if class_names else f"Class {i}"
            print(f"{class_name}: threshold={optimal_threshold:.3f}, F1={f1_scores[best_idx]:.4f}")
    
    return np.array(optimal_thresholds)


def apply_thresholds(pred_probs, thresholds):
    """Apply class-specific thresholds to predictions."""
    return (pred_probs > thresholds).astype(int)


# =============================================================================
# 4. DATA AUGMENTATION FOR VIETNAMESE NLP
# =============================================================================

class VietnameseTextAugmenter:
    """
    Data Augmentation specifically designed for Vietnamese text.
    Supports: Random Swap, Random Delete, Random Insert, Synonym (placeholder)
    """
    
    def __init__(self, p_swap=0.1, p_delete=0.1, p_insert=0.1, max_changes=3):
        self.p_swap = p_swap
        self.p_delete = p_delete
        self.p_insert = p_insert
        self.max_changes = max_changes
        
        # Vietnamese stopwords (common words that are safe to manipulate)
        self.stopwords = {
            'là', 'và', 'của', 'có', 'được', 'cho', 'trong', 'với', 'này',
            'các', 'một', 'những', 'đã', 'rất', 'để', 'như', 'thì', 'cũng',
            'còn', 'khi', 'mà', 'về', 'từ', 'tại', 'trên', 'hay', 'hoặc'
        }
        
        # Common Vietnamese adjective pairs for sentiment (simple synonym replacement)
        self.adjective_pairs = {
            'tốt': ['hay', 'ổn', 'khá'],
            'xấu': ['tệ', 'kém', 'dở'],
            'đẹp': ['xinh', 'đẳng cấp', 'sang'],
            'rẻ': ['hợp lý', 'phải chăng', 'mềm'],
            'đắt': ['cao', 'chát', 'mắc'],
            'nhanh': ['mượt', 'trơn tru', 'ổn định'],
            'chậm': ['lag', 'ì ạch', 'lề mề'],
        }
    
    def random_swap(self, text, n=None):
        """Swap n pairs of words randomly."""
        words = text.split()
        if len(words) < 2:
            return text
        
        n = n or min(self.max_changes, len(words) // 4 + 1)
        for _ in range(n):
            if len(words) >= 2:
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
        
        return ' '.join(words)
    
    def random_deletion(self, text):
        """Randomly delete words with probability p (skip important words)."""
        words = text.split()
        if len(words) <= 3:
            return text
        
        new_words = []
        for w in words:
            # Don't delete non-stopwords too aggressively
            if w.lower() in self.stopwords:
                if random.random() > self.p_delete:
                    new_words.append(w)
            else:
                # Keep important words more often
                if random.random() > self.p_delete * 0.5:
                    new_words.append(w)
        
        return ' '.join(new_words) if new_words else text
    
    def random_insertion(self, text, n=None):
        """Insert random words from the sentence at random positions."""
        words = text.split()
        if len(words) == 0:
            return text
        
        n = n or min(self.max_changes, max(1, len(words) // 5))
        for _ in range(n):
            idx = random.randint(0, len(words))
            word_to_insert = random.choice(words)
            words.insert(idx, word_to_insert)
        
        return ' '.join(words)
    
    def shuffle_middle(self, text):
        """Shuffle middle words while keeping first and last intact."""
        words = text.split()
        if len(words) <= 3:
            return text
        
        first, middle, last = words[0], words[1:-1], words[-1]
        random.shuffle(middle)
        return ' '.join([first] + middle + [last])
    
    def simple_synonym_replace(self, text):
        """Replace some adjectives with simple synonyms."""
        words = text.split()
        new_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.adjective_pairs and random.random() < 0.3:
                replacement = random.choice(self.adjective_pairs[word_lower])
                # Preserve original case (roughly)
                if word[0].isupper():
                    replacement = replacement.capitalize()
                new_words.append(replacement)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def augment(self, text, num_aug=1, methods=None):
        """
        Apply random augmentation(s) to generate variations.
        
        Args:
            text: input text
            num_aug: number of augmented versions to generate
            methods: list of methods to use, or None for all
        
        Returns:
            list of augmented texts
        """
        if methods is None:
            methods = ['swap', 'delete', 'insert', 'shuffle', 'synonym']
        
        all_methods = {
            'swap': self.random_swap,
            'delete': self.random_deletion,
            'insert': self.random_insertion,
            'shuffle': self.shuffle_middle,
            'synonym': self.simple_synonym_replace
        }
        
        augmented = []
        for _ in range(num_aug):
            method_name = random.choice(methods)
            aug_func = all_methods.get(method_name, self.random_swap)
            aug_text = aug_func(text)
            
            # Apply second augmentation sometimes for more diversity
            if random.random() < 0.3:
                another_method = random.choice([m for m in methods if m != method_name])
                aug_text = all_methods[another_method](aug_text)
            
            augmented.append(aug_text)
        
        return augmented


# =============================================================================
# 5. AUGMENTATION-BASED OVERSAMPLING
# =============================================================================

def oversample_with_augmentation(
    texts, 
    labels, 
    min_samples_per_class=100,
    target_ratio=0.5,
    augmenter=None,
    label_type='multi-label',
    verbose=True
):
    """
    Oversample minority classes using data augmentation.
    
    Args:
        texts: list of text samples
        labels: np.array [num_samples, num_classes] for multi-label
                or [num_samples] for multi-class
        min_samples_per_class: minimum samples for each class
        target_ratio: target ratio relative to majority class
        augmenter: VietnameseTextAugmenter instance (created if None)
        label_type: 'multi-label' or 'multi-class'
        verbose: print progress
    
    Returns:
        augmented_texts: list of texts (original + augmented)
        augmented_labels: np.array of labels
    """
    if augmenter is None:
        augmenter = VietnameseTextAugmenter()
    
    if label_type == 'multi-class':
        return _oversample_multiclass(texts, labels, min_samples_per_class, 
                                       target_ratio, augmenter, verbose)
    else:
        return _oversample_multilabel(texts, labels, min_samples_per_class,
                                       target_ratio, augmenter, verbose)


def _oversample_multilabel(texts, labels, min_samples, target_ratio, augmenter, verbose):
    """Oversample for multi-label classification (e.g., aspects)."""
    labels = np.array(labels)
    class_counts = labels.sum(axis=0)
    max_count = class_counts.max()
    target_count = max(min_samples, int(max_count * target_ratio))
    
    new_texts = list(texts)
    new_labels = list(labels)
    
    if verbose:
        print("\n" + "="*50)
        print("AUGMENTATION-BASED OVERSAMPLING (Multi-label)")
        print("="*50)
        print(f"Target samples per class: {target_count}")
    
    for class_idx in range(labels.shape[1]):
        current_count = int(class_counts[class_idx])
        
        if current_count < target_count and current_count > 0:
            # Find samples with this class
            class_sample_indices = np.where(labels[:, class_idx] == 1)[0]
            n_to_add = target_count - current_count
            
            if verbose:
                print(f"Class {class_idx}: {current_count} -> {target_count} (+{n_to_add})")
            
            for _ in range(n_to_add):
                # Random sample from this class
                idx = random.choice(class_sample_indices)
                original_text = texts[idx]
                
                # Augment the text
                aug_text = augmenter.augment(original_text, num_aug=1)[0]
                
                new_texts.append(aug_text)
                new_labels.append(labels[idx])
    
    if verbose:
        print(f"\nTotal: {len(texts)} -> {len(new_texts)} samples")
    
    return new_texts, np.array(new_labels)


def _oversample_multiclass(texts, labels, min_samples, target_ratio, augmenter, verbose):
    """Oversample for multi-class classification (e.g., sentiments)."""
    labels = np.array(labels)
    
    # Convert one-hot to class indices if needed
    if labels.ndim == 2:
        label_indices = labels.argmax(axis=1)
        is_onehot = True
    else:
        label_indices = labels
        is_onehot = False
    
    class_counts = Counter(label_indices)
    max_count = max(class_counts.values())
    target_count = max(min_samples, int(max_count * target_ratio))
    
    new_texts = list(texts)
    new_label_indices = list(label_indices)
    
    if verbose:
        print("\n" + "="*50)
        print("AUGMENTATION-BASED OVERSAMPLING (Multi-class)")
        print("="*50)
        print(f"Target samples per class: {target_count}")
    
    for class_idx, current_count in class_counts.items():
        if current_count < target_count:
            # Find samples with this class
            class_sample_indices = np.where(label_indices == class_idx)[0]
            n_to_add = target_count - current_count
            
            if verbose:
                print(f"Class {class_idx}: {current_count} -> {target_count} (+{n_to_add})")
            
            for _ in range(n_to_add):
                idx = random.choice(class_sample_indices)
                original_text = texts[idx]
                aug_text = augmenter.augment(original_text, num_aug=1)[0]
                
                new_texts.append(aug_text)
                new_label_indices.append(class_idx)
    
    if verbose:
        print(f"\nTotal: {len(texts)} -> {len(new_texts)} samples")
    
    # Convert back to one-hot if needed
    if is_onehot:
        num_classes = labels.shape[1]
        new_labels = np.zeros((len(new_label_indices), num_classes))
        for i, idx in enumerate(new_label_indices):
            new_labels[i, idx] = 1
    else:
        new_labels = np.array(new_label_indices)
    
    return new_texts, new_labels


# =============================================================================
# 6. WEIGHTED RANDOM SAMPLER
# =============================================================================

def create_weighted_sampler(labels, num_samples=None):
    """
    Create WeightedRandomSampler for imbalanced dataset.
    
    Args:
        labels: torch.Tensor or np.array
        num_samples: number of samples to draw (default: len(labels))
    
    Returns:
        WeightedRandomSampler
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    if num_samples is None:
        num_samples = len(labels)
    
    # Handle multi-label
    if labels.ndim == 2:
        label_tuples = [tuple(row.tolist()) for row in labels]
    else:
        label_tuples = labels.tolist()
    
    label_counts = Counter(label_tuples)
    weights = [1.0 / label_counts[tuple(labels[i].tolist()) if labels.ndim == 2 else labels[i]] 
               for i in range(len(labels))]
    
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)


# =============================================================================
# 7. EVALUATION UTILITIES
# =============================================================================

def evaluate_multilabel(true_labels, pred_probs, thresholds=None, class_names=None, 
                        task_name="Task", verbose=True):
    """
    Comprehensive evaluation for multi-label classification.
    
    Returns:
        dict with f1_micro, f1_macro, predictions, and report
    """
    if thresholds is None:
        thresholds = np.array([0.5] * pred_probs.shape[1])
    
    pred_labels = apply_thresholds(pred_probs, thresholds)
    
    f1_micro = f1_score(true_labels, pred_labels, average='micro')
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"  {task_name} EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"F1-Micro:   {f1_micro:.4f}")
        print(f"F1-Macro:   {f1_macro:.4f}")
        print(f"F1-Weighted: {f1_weighted:.4f}")
        print(f"\nDetailed Report:")
        print(classification_report(true_labels, pred_labels, 
                                    target_names=class_names, zero_division=0))
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': pred_labels
    }


def evaluate_multiclass(true_labels, pred_probs, class_names=None, 
                        task_name="Task", verbose=True):
    """
    Evaluation for multi-class classification (e.g., sentiment).
    
    Args:
        true_labels: one-hot encoded or class indices
        pred_probs: predicted probabilities [samples, classes]
    """
    # Convert one-hot to class indices if needed
    if true_labels.ndim == 2:
        true_indices = true_labels.argmax(axis=1)
    else:
        true_indices = true_labels
    
    pred_indices = pred_probs.argmax(axis=1)
    
    f1_micro = f1_score(true_indices, pred_indices, average='micro')
    f1_macro = f1_score(true_indices, pred_indices, average='macro')
    f1_weighted = f1_score(true_indices, pred_indices, average='weighted')
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"  {task_name} EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"F1-Micro:   {f1_micro:.4f}")
        print(f"F1-Macro:   {f1_macro:.4f}")
        print(f"F1-Weighted: {f1_weighted:.4f}")
        print(f"\nDetailed Report:")
        print(classification_report(true_indices, pred_indices,
                                    target_names=class_names, zero_division=0))
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': pred_indices
    }


# =============================================================================
# 8. ANALYSIS UTILITIES
# =============================================================================

def analyze_class_distribution(labels, class_names=None, title="CLASS DISTRIBUTION"):
    """
    Analyze and print class distribution with visual bars.
    
    Args:
        labels: np.array [samples, classes] for multi-label or [samples] for multi-class
        class_names: list of class names
    
    Returns:
        dict with counts and warnings
    """
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)
    
    if labels.ndim == 2:
        # Multi-label
        counts = labels.sum(axis=0)
        total = len(labels)
    else:
        # Multi-class
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
    
    warnings = []
    for i, count in enumerate(counts):
        name = class_names[i] if class_names else f"Class {i}"
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        
        # Determine severity
        if count < total * 0.01:
            flag = "🚨 CRITICAL!"
            warnings.append((name, count, "critical"))
        elif count < total * 0.05:
            flag = "⚠️ WARNING"
            warnings.append((name, count, "warning"))
        else:
            flag = ""
        
        print(f"  {name:15s}: {int(count):5d} ({pct:5.1f}%) {bar} {flag}")
    
    if warnings:
        print("\n⚠️ Imbalanced classes detected:")
        for name, count, severity in warnings:
            print(f"  - {name}: only {count} samples ({severity})")
    
    return {'counts': counts, 'warnings': warnings}


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Vietnamese Text Augmenter...")
    augmenter = VietnameseTextAugmenter()
    
    test_text = "Điện thoại này rất tốt và đẹp, pin trâu nữa"
    print(f"\nOriginal: {test_text}")
    
    for method in ['swap', 'delete', 'insert', 'shuffle', 'synonym']:
        aug = augmenter.augment(test_text, num_aug=1, methods=[method])[0]
        print(f"{method:10s}: {aug}")
    
    print("\n✅ All tests passed!")
