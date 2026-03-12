"""
FIXED EVALUATION CODE - Copy this into your notebook

The fix: Convert arrays to boolean before using bitwise operators (&)
"""

# ============================================================
# COPY THIS FIXED CODE INTO YOUR NOTEBOOK
# ============================================================

FIXED_EVALUATION_CODE = '''
# ============================================================
# THRESHOLD TUNING & FINAL EVALUATION (IMPROVED)
# ============================================================

results = {}

for name, model in models.items():
    print(f"\\n{'='*60}")
    print(f"📊 Evaluating {name}")
    print(f"{'='*60}")
    
    # Dev for threshold tuning
    dev_res = evaluate(model, dev_loader)
    if USE_THRESHOLD_TUNING:
        thresholds = find_thresholds(dev_res['asp_true'], dev_res['asp_probs'], ASPECT_NAMES)
    else:
        thresholds = np.array([0.5] * NUM_ASPECTS)
    
    # Test evaluation
    test_res = evaluate(model, test_loader)
    
    # ==============================================================
    # ASPECT EVALUATION (Multi-label)
    # ==============================================================
    asp_default = (test_res['asp_probs'] > 0.5).astype(int)
    asp_optimal = (test_res['asp_probs'] > thresholds).astype(int)
    
    f1_asp_default = f1_score(test_res['asp_true'], asp_default, average='micro')
    f1_asp_optimal = f1_score(test_res['asp_true'], asp_optimal, average='micro')
    f1_asp_macro = f1_score(test_res['asp_true'], asp_optimal, average='macro')
    
    # ==============================================================
    # SENTIMENT EVALUATION
    # ==============================================================
    sent_pred = np.argmax(test_res['sent_probs'], axis=1)
    sent_true = np.argmax(test_res['sent_true'], axis=1)
    f1_sent_micro = f1_score(sent_true, sent_pred, average='micro')
    f1_sent_macro = f1_score(sent_true, sent_pred, average='macro')
    
    # Sample distribution
    from collections import Counter
    true_counts = Counter(sent_true)
    pred_counts = Counter(sent_pred)
    
    results[name] = {
        'asp_default': f1_asp_default,
        'asp_optimal': f1_asp_optimal,
        'asp_macro': f1_asp_macro,
        'sent_micro': f1_sent_micro,
        'sent_macro': f1_sent_macro,
        'thresholds': thresholds
    }
    
    # ========== ASPECT RESULTS ==========
    print(f"\\n🏷️ ASPECT EXTRACTION RESULTS:")
    print(f"   F1-Micro (default 0.5): {f1_asp_default:.4f}")
    print(f"   F1-Micro (optimized):   {f1_asp_optimal:.4f} (+{(f1_asp_optimal-f1_asp_default)*100:.2f}%)")
    print(f"   F1-Macro:               {f1_asp_macro:.4f}")
    
    # Per-aspect breakdown (FIXED: convert to bool)
    print(f"\\n   Per-Aspect Performance:")
    print(f"   {'Aspect':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"   {'-'*55}")
    for i, asp_name in enumerate(ASPECT_NAMES):
        y_true = test_res['asp_true'][:, i].astype(bool)  # FIX: convert to bool
        y_pred = asp_optimal[:, i].astype(bool)           # FIX: convert to bool
        support = int(y_true.sum())
        if support > 0:
            tp = (y_true & y_pred).sum()
            prec = tp / max(1, y_pred.sum())
            rec = tp / max(1, support)
            f1 = 2 * prec * rec / max(0.0001, prec + rec)
        else:
            prec, rec, f1 = 0, 0, 0
        print(f"   {asp_name:<12} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {support:>10}")
    
    # ========== SENTIMENT RESULTS ==========
    print(f"\\n💭 SENTIMENT CLASSIFICATION RESULTS:")
    print(f"   F1-Micro (Accuracy): {f1_sent_micro:.4f}")
    print(f"   F1-Macro:            {f1_sent_macro:.4f}")
    
    # Sample distribution analysis
    print(f"\\n   Sample Distribution (Ground Truth vs Predictions):")
    print(f"   {'Class':<12} {'True Count':>12} {'Pred Count':>12}")
    print(f"   {'-'*38}")
    for i, sent_name in enumerate(SENTIMENT_NAMES):
        tc = true_counts.get(i, 0)
        pc = pred_counts.get(i, 0)
        flag = '⚠️' if tc < 50 else ''
        print(f"   {sent_name:<12} {tc:>12} {pc:>12} {flag}")
    
    # Per-class sentiment metrics
    print(f"\\n   Per-Class Performance:")
    print(f"   {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"   {'-'*55}")
    for i, sent_name in enumerate(SENTIMENT_NAMES):
        mask_true = (sent_true == i)  # Already boolean
        mask_pred = (sent_pred == i)  # Already boolean
        tp = (mask_true & mask_pred).sum()
        fp = (~mask_true & mask_pred).sum()
        fn = (mask_true & ~mask_pred).sum()
        
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(0.0001, prec + rec)
        support = mask_true.sum()
        
        flag = '🔴' if support < 20 else ''
        print(f"   {sent_name:<12} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {support:>10} {flag}")
    
    # Insight for very low support
    neg_support = true_counts.get(1, 0)
    if neg_support < 50:
        print(f"\\n   ⚠️ WARNING: NEGATIVE has only {neg_support} samples. Metrics may be unreliable.")

print(f"\\n\\n{'='*60}")
print(f"📊 FINAL COMPARISON SUMMARY")
print(f"{'='*60}\\n")

print(f"{'Model':<10} {'Asp F1-Micro':>14} {'Asp F1-Macro':>14} {'Sent F1-Micro':>14} {'Sent F1-Macro':>14}")
print("-" * 70)
for name, res in results.items():
    print(f"{name:<10} {res['asp_optimal']:>14.4f} {res['asp_macro']:>14.4f} "
          f"{res['sent_micro']:>14.4f} {res['sent_macro']:>14.4f}")

best_asp = max(results.keys(), key=lambda x: results[x]['asp_optimal'])
best_sent = max(results.keys(), key=lambda x: results[x]['sent_micro'])
print(f"\\n🏆 Best Aspect Model:    {best_asp} (F1-Micro = {results[best_asp]['asp_optimal']:.4f})")
print(f"🏆 Best Sentiment Model: {best_sent} (F1-Micro = {results[best_sent]['sent_micro']:.4f})")

print(f"\\n🔧 Techniques Used:")
print(f"  ├── Focal Loss:         {'✅' if USE_FOCAL_LOSS else '❌'}")
print(f"  ├── Class Weighting:    {'✅' if USE_CLASS_WEIGHTS else '❌'}")
print(f"  ├── Threshold Tuning:   {'✅' if USE_THRESHOLD_TUNING else '❌'}")
print(f"  └── Augmentation:       {'✅' if len(train_ds) > 7000 else '❌'}")
'''

if __name__ == "__main__":
    print(FIXED_EVALUATION_CODE)
