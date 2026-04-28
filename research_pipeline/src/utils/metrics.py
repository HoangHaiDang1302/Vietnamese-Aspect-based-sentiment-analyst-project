"""
Metrics & Evaluation Module
Chứa toàn bộ hàm đánh giá cho cả 3 task: ATE, ASC, E2E
Được kế thừa từ baseline_all_models_crf.ipynb
"""
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# ============================================================
# 1. BIO SPAN EXTRACTION (Dùng cho ATE và E2E)
# ============================================================
def bio_tags_to_spans(tag_ids, bio_tags_list, max_tokens):
    """
    Chuyển chuỗi BIO tag IDs -> danh sách spans (label, start, end).
    Ví dụ: [O, B-CAMERA, I-CAMERA, O] -> [("CAMERA", 1, 3)]
    """
    spans = []
    current_label = None
    current_start = None
    for t in range(min(len(tag_ids), max_tokens)):
        tag_id = tag_ids[t]
        tag_name = bio_tags_list[tag_id] if tag_id < len(bio_tags_list) else 'O'
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


# ============================================================
# 1b. CHARACTER SPANS → WORD SPANS (Dùng cho Pipeline eval)
# ============================================================
def char_spans_to_word_spans(text, raw_labels, max_len):
    """
    Chuyển character-level ground truth spans → word-level spans.
    Input:  raw_labels = [(start_char, end_char, "CAMERA#POSITIVE"), ...]
    Output: [("CAMERA#POSITIVE", start_word, end_word), ...]
    Dùng để so sánh span-level với predictions từ Pipeline (NB03).
    """
    words = text.split()[:max_len]
    positions = []
    pos = 0
    text_lower = text.lower()

    for w in words:
        w_clean = w.replace('_', ' ')
        idx = text_lower.find(w_clean, pos)
        if idx == -1:
            idx = pos
        positions.append((idx, idx + len(w_clean)))
        pos = idx + len(w_clean)

    spans = []
    for start_char, end_char, raw_label in raw_labels:
        word_indices = []
        for t_idx in range(len(words)):
            t_start, t_end = positions[t_idx]
            if t_start < end_char and t_end > start_char:
                word_indices.append(t_idx)
        if word_indices:
            spans.append((raw_label, word_indices[0], word_indices[-1] + 1))

    return spans


# ============================================================
# 2. SPAN-LEVEL F1 (Chính xác cho ATE)
# ============================================================
def evaluate_spans_f1(pred_spans_list, true_spans_list):
    """
    Span-level Precision / Recall / F1 cho ATE.
    Một span được coi là đúng nếu trùng khớp hoàn toàn (label, start, end).
    """
    tp, fp, fn = 0, 0, 0
    for pred_spans, true_spans in zip(pred_spans_list, true_spans_list):
        pred_set = set((l, s, e) for l, s, e in pred_spans)
        true_set = set((l, s, e) for l, s, e in true_spans)
        tp += len(pred_set & true_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {'precision': p, 'recall': r, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


# ============================================================
# 2b. BIO TAGS → SENTENCE-LEVEL MULTI-LABEL (Dùng cho tất cả notebooks)
# ============================================================
def bio_to_sentence_labels(tag_ids_list, lengths, bio_tags_list, label_names):
    """
    Chuyển BIO tag sequences → sentence-level multi-label binary vectors.
    Giống cách phobert-crf-absa.ipynb đánh giá.

    Args:
        tag_ids_list: list of tag_id sequences (per sentence)
        lengths: list of actual lengths
        bio_tags_list: danh sách BIO tag names (vd ['O','B-CAMERA#POS',...])
        label_names: danh sách label names (vd ['CAMERA#POSITIVE',...])
    Returns:
        sent_labels: np.array (N, num_labels) binary matrix
        all_spans: list of span lists per sentence
    """
    label2id = {ln: i for i, ln in enumerate(label_names)}
    num_labels = len(label_names)
    sent_labels, all_spans = [], []

    for tag_ids, length in zip(tag_ids_list, lengths):
        spans = bio_tags_to_spans(tag_ids, bio_tags_list, length)
        label_vec = [0] * num_labels
        for label_name, _, _ in spans:
            if label_name in label2id:
                label_vec[label2id[label_name]] = 1
        sent_labels.append(label_vec)
        all_spans.append(spans)

    return np.array(sent_labels), all_spans


# ============================================================
# 3. MULTI-LABEL EVALUATION (Dùng cho E2E sentence-level)
# ============================================================
def evaluate_multilabel(y_true, y_pred, label_names):
    """
    Đánh giá Multi-label (kế thừa nguyên từ baseline_all_models_crf).
    Trả về dict chứa per-label metrics + micro/macro/weighted.
    """
    y_true, y_pred = np.array(y_true).astype(int), np.array(y_pred).astype(int)
    results = {}
    all_p, all_r, all_f1, all_sup = [], [], [], []
    for i in range(y_true.shape[1]):
        tp = int(((y_true[:, i] == 1) & (y_pred[:, i] == 1)).sum())
        fp = int(((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum())
        fn = int(((y_true[:, i] == 1) & (y_pred[:, i] == 0)).sum())
        sup = int(y_true[:, i].sum())
        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f = 2 * p * r / (p + r) if p + r > 0 else 0
        results[label_names[i]] = {'precision': p, 'recall': r, 'f1': f, 'support': sup}
        all_p.append(p); all_r.append(r); all_f1.append(f); all_sup.append(sup)

    results['macro'] = {'precision': np.mean(all_p), 'recall': np.mean(all_r), 'f1': np.mean(all_f1)}
    ts = sum(all_sup)
    if ts > 0:
        results['weighted'] = {
            'precision': sum(p * s for p, s in zip(all_p, all_sup)) / ts,
            'recall': sum(r * s for r, s in zip(all_r, all_sup)) / ts,
            'f1': sum(f * s for f, s in zip(all_f1, all_sup)) / ts
        }
    # Micro
    ttp = sum(int(((y_true[:, i] == 1) & (y_pred[:, i] == 1)).sum()) for i in range(y_true.shape[1]))
    tfp = sum(int(((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum()) for i in range(y_true.shape[1]))
    tfn = sum(int(((y_true[:, i] == 1) & (y_pred[:, i] == 0)).sum()) for i in range(y_true.shape[1]))
    mp = ttp / (ttp + tfp) if ttp + tfp > 0 else 0
    mr = ttp / (ttp + tfn) if ttp + tfn > 0 else 0
    results['micro'] = {'precision': mp, 'recall': mr, 'f1': 2 * mp * mr / (mp + mr) if mp + mr > 0 else 0}
    return results


# ============================================================
# 4. ASC CLASSIFICATION METRICS
# ============================================================
def evaluate_asc(y_true, y_pred, class_names=None):
    """
    Đánh giá Classification (Accuracy, Macro-F1, Weighted-F1) cho ASC.
    """
    if class_names is None:
        class_names = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'report': report,
        'confusion_matrix': cm
    }


# ============================================================
# 5. TOKEN-LEVEL ACCURACY (Hỗ trợ debug Sequence Labeling)
# ============================================================
def token_accuracy(pred_tags_list, true_tags_list):
    """Tính phần trăm token đúng (bỏ qua PAD)"""
    correct, total = 0, 0
    for pt, tt in zip(pred_tags_list, true_tags_list):
        for p, t in zip(pt, tt):
            if p == t:
                correct += 1
            total += 1
    return correct / max(1, total)
