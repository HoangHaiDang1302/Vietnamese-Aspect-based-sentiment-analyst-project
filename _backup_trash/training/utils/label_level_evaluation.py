"""
Label-Level Evaluation cho Multi-Label Classification
======================================================

Đánh giá hiệu năng mô hình theo Label-Level (từng nhãn độc lập)
cho bài toán Aspect-Based Sentiment Analysis.

Trong đánh giá Label-Level:
- Mỗi nhãn (aspect#sentiment) được đánh giá độc lập
- Cho mỗi nhãn, tính TP, FP, FN dựa trên so sánh từng cột của ma trận multi-hot
- Precision = TP / (TP + FP): Trong các dự đoán positive, bao nhiêu % đúng
- Recall = TP / (TP + FN): Trong các ground truth positive, bao nhiêu % được dự đoán đúng
- F1 = 2 * P * R / (P + R)
"""

import numpy as np
from typing import List, Dict, Optional, Union


def evaluate_label_level(
    y_true: Union[List[List[int]], np.ndarray],
    y_pred: Union[List[List[int]], np.ndarray],
    label_names: Optional[List[str]] = None,
    average: Optional[str] = 'macro',
    print_report: bool = False
) -> Dict:
    """
    Đánh giá hiệu năng mô hình theo Label-Level cho bài toán multi-label classification.
    
    Cách tính đúng cho multi-label:
    - Với mỗi label (cột), tính TP, FP, FN giữa y_true[:, i] và y_pred[:, i]
    - Precision_i = TP_i / (TP_i + FP_i)
    - Recall_i = TP_i / (TP_i + FN_i)
    - F1_i = 2 * P_i * R_i / (P_i + R_i)
    
    Args:
        y_true: Ma trận ground truth multi-hot, shape (num_samples, num_labels)
        y_pred: Ma trận dự đoán multi-hot, shape (num_samples, num_labels)
        label_names: Danh sách tên nhãn để hiển thị trong báo cáo
        average: Cách tính trung bình:
            - 'micro': Tính tổng global TP, FP, FN rồi tính P, R, F1
            - 'macro': Trung bình đơn giản của P, R, F1 của từng label
            - 'weighted': Trung bình có trọng số theo support của từng label
            - None: Trả về metrics cho từng label
        print_report: In báo cáo chi tiết nếu True
    
    Returns:
        dict: Dictionary chứa các chỉ số đánh giá label-level
    """
    # Convert to numpy arrays
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    # Validate shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    num_samples, num_labels = y_true.shape
    
    # Tính metrics cho từng label (cột)
    label_metrics = []
    
    for i in range(num_labels):
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]
        
        # Tính TP, FP, FN, TN
        tp = np.sum((true_col == 1) & (pred_col == 1))
        fp = np.sum((true_col == 0) & (pred_col == 1))
        fn = np.sum((true_col == 1) & (pred_col == 0))
        tn = np.sum((true_col == 0) & (pred_col == 0))
        
        # Support: số lượng positive trong ground truth
        support = int(np.sum(true_col))
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        label_metrics.append({
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'support': support,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Tính average metrics
    if average == 'micro':
        # Tổng hợp global TP, FP, FN
        total_tp = sum(m['tp'] for m in label_metrics)
        total_fp = sum(m['fp'] for m in label_metrics)
        total_fn = sum(m['fn'] for m in label_metrics)
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) \
                   if (micro_precision + micro_recall) > 0 else 0.0
        
        results = {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1_score': micro_f1,
            'average': 'micro',
            'num_labels': num_labels,
            'num_samples': num_samples
        }
        
    elif average == 'macro':
        # Trung bình đơn giản
        macro_precision = np.mean([m['precision'] for m in label_metrics])
        macro_recall = np.mean([m['recall'] for m in label_metrics])
        macro_f1 = np.mean([m['f1'] for m in label_metrics])
        
        results = {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1_score': macro_f1,
            'average': 'macro',
            'num_labels': num_labels,
            'num_samples': num_samples
        }
        
    elif average == 'weighted':
        # Trung bình có trọng số theo support
        total_support = sum(m['support'] for m in label_metrics)
        
        if total_support > 0:
            weighted_precision = sum(m['precision'] * m['support'] for m in label_metrics) / total_support
            weighted_recall = sum(m['recall'] * m['support'] for m in label_metrics) / total_support
            weighted_f1 = sum(m['f1'] * m['support'] for m in label_metrics) / total_support
        else:
            weighted_precision = weighted_recall = weighted_f1 = 0.0
        
        results = {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1_score': weighted_f1,
            'average': 'weighted',
            'num_labels': num_labels,
            'num_samples': num_samples,
            'total_support': total_support
        }
        
    else:  # average is None - return per-label metrics
        per_label_results = {}
        for i, metrics in enumerate(label_metrics):
            if label_names and i < len(label_names):
                label_key = label_names[i]
            else:
                label_key = f"label_{i}"
            
            per_label_results[label_key] = {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1'],
                'support': metrics['support'],
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn']
            }
        
        # Thêm summary statistics
        per_label_results['_summary'] = {
            'macro_precision': np.mean([m['precision'] for m in label_metrics]),
            'macro_recall': np.mean([m['recall'] for m in label_metrics]),
            'macro_f1': np.mean([m['f1'] for m in label_metrics]),
            'num_labels': num_labels,
            'num_samples': num_samples
        }
        
        results = per_label_results
    
    # In báo cáo chi tiết nếu cần
    if print_report:
        print_label_level_report(label_metrics, label_names, average)
    
    return results


def print_label_level_report(
    label_metrics: List[Dict],
    label_names: Optional[List[str]] = None,
    average: Optional[str] = 'macro'
):
    """
    In báo cáo đánh giá label-level theo định dạng đẹp.
    """
    print("\n" + "=" * 70)
    print("[REPORT] LABEL-LEVEL EVALUATION REPORT")
    print("=" * 70)
    
    # Header
    header = f"{'Label':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    print(f"\n{header}")
    print("-" * 70)
    
    # Per-label metrics
    for i, metrics in enumerate(label_metrics):
        if label_names and i < len(label_names):
            label = label_names[i]
        else:
            label = f"label_{i}"
        
        # Truncate long names
        if len(label) > 24:
            label = label[:21] + "..."
        
        print(f"{label:<25} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['support']:>10d}")
    
    print("-" * 70)
    
    # Average metrics
    if average:
        if average == 'micro':
            total_tp = sum(m['tp'] for m in label_metrics)
            total_fp = sum(m['fp'] for m in label_metrics)
            total_fn = sum(m['fn'] for m in label_metrics)
            
            p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            print(f"{'micro avg':<25} {p:>10.4f} {r:>10.4f} {f1:>10.4f} "
                  f"{sum(m['support'] for m in label_metrics):>10d}")
        
        elif average == 'macro':
            p = np.mean([m['precision'] for m in label_metrics])
            r = np.mean([m['recall'] for m in label_metrics])
            f1 = np.mean([m['f1'] for m in label_metrics])
            
            print(f"{'macro avg':<25} {p:>10.4f} {r:>10.4f} {f1:>10.4f} "
                  f"{sum(m['support'] for m in label_metrics):>10d}")
        
        elif average == 'weighted':
            total_support = sum(m['support'] for m in label_metrics)
            if total_support > 0:
                p = sum(m['precision'] * m['support'] for m in label_metrics) / total_support
                r = sum(m['recall'] * m['support'] for m in label_metrics) / total_support
                f1 = sum(m['f1'] * m['support'] for m in label_metrics) / total_support
            else:
                p = r = f1 = 0.0
            
            print(f"{'weighted avg':<25} {p:>10.4f} {r:>10.4f} {f1:>10.4f} "
                  f"{total_support:>10d}")
    
    print()


def evaluate_aspect_sentiment_combined(
    aspect_true: np.ndarray,
    aspect_pred: np.ndarray,
    sentiment_true: np.ndarray,
    sentiment_pred: np.ndarray,
    aspect_names: List[str],
    sentiment_names: List[str],
    average: str = 'macro',
    print_report: bool = True
) -> Dict:
    """
    Đánh giá kết hợp Aspect và Sentiment theo label-level.
    Tạo combined labels 30-D (10 aspects x 3 sentiments) và đánh giá.
    
    Args:
        aspect_true: Ground truth aspects, shape (N, 10)
        aspect_pred: Predicted aspects, shape (N, 10)
        sentiment_true: Ground truth sentiments (multi-hot 3-D), shape (N, 3)
        sentiment_pred: Predicted sentiments (multi-hot 3-D), shape (N, 3)
        aspect_names: List of aspect names
        sentiment_names: List of sentiment names
        average: 'micro', 'macro', 'weighted', or None
        print_report: Print detailed report if True
    
    Returns:
        dict: Label-level evaluation metrics
    """
    num_samples = aspect_true.shape[0]
    num_aspects = len(aspect_names)
    num_sentiments = len(sentiment_names)
    num_combined = num_aspects * num_sentiments  # 30
    
    # Tạo combined labels (30-D)
    # Label i*3 + j tương ứng với aspect i và sentiment j
    combined_true = np.zeros((num_samples, num_combined), dtype=int)
    combined_pred = np.zeros((num_samples, num_combined), dtype=int)
    
    for sample_idx in range(num_samples):
        for asp_idx in range(num_aspects):
            for sent_idx in range(num_sentiments):
                combined_idx = asp_idx * num_sentiments + sent_idx
                
                # Combined label = 1 nếu cả aspect và sentiment đều = 1
                combined_true[sample_idx, combined_idx] = int(
                    aspect_true[sample_idx, asp_idx] == 1 and 
                    sentiment_true[sample_idx, sent_idx] == 1
                )
                combined_pred[sample_idx, combined_idx] = int(
                    aspect_pred[sample_idx, asp_idx] == 1 and 
                    sentiment_pred[sample_idx, sent_idx] == 1
                )
    
    # Tạo tên cho combined labels
    combined_names = []
    for asp_name in aspect_names:
        for sent_name in sentiment_names:
            combined_names.append(f"{asp_name}#{sent_name}")
    
    # Đánh giá label-level
    results = evaluate_label_level(
        y_true=combined_true,
        y_pred=combined_pred,
        label_names=combined_names,
        average=average,
        print_report=print_report
    )
    
    return results


# ============================================================
# VÍ DỤ SỬ DỤNG TRONG NOTEBOOK
# ============================================================

if __name__ == "__main__":
    # Ví dụ với dữ liệu giả
    np.random.seed(42)
    
    # Giả lập dữ liệu multi-label (100 mẫu, 10 labels)
    y_true = np.random.randint(0, 2, size=(100, 10))
    y_pred = np.random.randint(0, 2, size=(100, 10))
    
    label_names = [f"Label_{i}" for i in range(10)]
    
    print("=" * 70)
    print("TEST LABEL-LEVEL EVALUATION")
    print("=" * 70)
    
    # Test micro average
    print("\n[*] Micro Average:")
    micro_results = evaluate_label_level(y_true, y_pred, label_names, average='micro')
    print(f"  Precision: {micro_results['precision']:.4f}")
    print(f"  Recall: {micro_results['recall']:.4f}")
    print(f"  F1-Score: {micro_results['f1_score']:.4f}")
    
    # Test macro average
    print("\n[*] Macro Average:")
    macro_results = evaluate_label_level(y_true, y_pred, label_names, average='macro')
    print(f"  Precision: {macro_results['precision']:.4f}")
    print(f"  Recall: {macro_results['recall']:.4f}")
    print(f"  F1-Score: {macro_results['f1_score']:.4f}")
    
    # Test weighted average
    print("\n[*] Weighted Average:")
    weighted_results = evaluate_label_level(y_true, y_pred, label_names, average='weighted')
    print(f"  Precision: {weighted_results['precision']:.4f}")
    print(f"  Recall: {weighted_results['recall']:.4f}")
    print(f"  F1-Score: {weighted_results['f1_score']:.4f}")
    
    # Test per-label với báo cáo chi tiết
    print("\n[*] Per-Label Details (with report):")
    per_label_results = evaluate_label_level(
        y_true, y_pred, label_names, 
        average=None, 
        print_report=True
    )
