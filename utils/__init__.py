"""
Utility modules for Vietnamese ABSA project.
Version 3.0 - Complete utilities with text processing and imbalance handling
"""

from .imbalance_utils import (
    # Loss Functions
    FocalLoss,
    FocalLossMultiClass,
    LabelSmoothingBCELoss,
    
    # Class Weights
    calculate_class_weights,
    calculate_pos_weight_for_bce,
    
    # Threshold Tuning
    find_optimal_thresholds,
    apply_thresholds,
    
    # Data Augmentation & Oversampling
    VietnameseTextAugmenter,
    oversample_with_augmentation,
    
    # Sampling
    create_weighted_sampler,
    
    # Evaluation
    evaluate_multilabel,
    evaluate_multiclass,
    analyze_class_distribution
)

__all__ = [
    # Loss Functions
    'FocalLoss',
    'FocalLossMultiClass', 
    'LabelSmoothingBCELoss',
    
    # Class Weights
    'calculate_class_weights',
    'calculate_pos_weight_for_bce',
    
    # Threshold Tuning
    'find_optimal_thresholds',
    'apply_thresholds',
    
    # Data Augmentation & Oversampling
    'VietnameseTextAugmenter',
    'oversample_with_augmentation',
    
    # Sampling
    'create_weighted_sampler',
    
    # Evaluation
    'evaluate_multilabel',
    'evaluate_multiclass',
    'analyze_class_distribution'
]

# Text processing utilities
from .text_utils import (
    # Text Cleaning
    normalize_unicode,
    clean_text,
    segment_vietnamese,
    preprocess_text,
    batch_preprocess,
    
    # Augmentation
    TextAugmenter
)

__all__ += [
    # Text Processing
    'normalize_unicode',
    'clean_text',
    'segment_vietnamese',
    'preprocess_text',
    'batch_preprocess',
    'TextAugmenter'
]
