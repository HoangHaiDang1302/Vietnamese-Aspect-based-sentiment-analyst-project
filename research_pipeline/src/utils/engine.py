"""
Training Engine Module
Chứa toàn bộ vòng lặp huấn luyện, dự đoán, early stopping.
Kế thừa từ baseline_all_models_crf.ipynb và cải tiến thêm.
"""
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm


# ============================================================
# 1. ATE TRAINING ENGINE (Sequence Labeling + CRF)
# ============================================================
def train_epoch_ate(model, loader, optimizer, device):
    """Một epoch huấn luyện cho ATE (CRF-based)"""
    model.train()
    total_loss = 0
    for batch in tqdm(loader, leave=False, desc="Training"):
        optimizer.zero_grad()
        seqs = batch['seq'].to(device)
        masks = batch['mask'].to(device)
        tags = batch['tags'].to(device)
        lens = batch['len']

        loss = model(seqs, mask=masks, labels=tags, lens=lens)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def predict_ate(model, loader, device, bio_tags_list=None):
    """Dự đoán toàn bộ tập dữ liệu cho ATE. Trả về pred_tags, true_tags, lengths, span_f1"""
    from src.utils.metrics import bio_tags_to_spans, evaluate_spans_f1

    model.eval()
    all_pred_tags, all_true_tags, all_lengths = [], [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            seqs = batch['seq'].to(device)
            masks = batch['mask'].to(device)
            tags = batch['tags'].to(device)
            lens = batch['len']

            # Loss
            loss = model(seqs, mask=masks, labels=tags, lens=lens)
            total_loss += loss.item()

            # Decode
            pred_tags = model(seqs, mask=masks, lens=lens)

            for i in range(len(lens)):
                length = lens[i].item()
                all_pred_tags.append(pred_tags[i][:length])
                all_true_tags.append(tags[i][:length].cpu().tolist())
                all_lengths.append(length)

    # Token accuracy
    correct, total = 0, 0
    for pt, tt in zip(all_pred_tags, all_true_tags):
        for p, t in zip(pt, tt):
            if p == t: correct += 1
            total += 1
    tok_acc = correct / max(1, total)

    # Span F1
    span_f1 = 0.0
    if bio_tags_list is not None:
        ps = [bio_tags_to_spans(pt, bio_tags_list, l) for pt, l in zip(all_pred_tags, all_lengths)]
        ts = [bio_tags_to_spans(tt, bio_tags_list, l) for tt, l in zip(all_true_tags, all_lengths)]
        f1_res = evaluate_spans_f1(ps, ts)
        span_f1 = f1_res['f1']

    return {
        'loss': total_loss / len(loader),
        'pred_tags': all_pred_tags,
        'true_tags': all_true_tags,
        'lengths': all_lengths,
        'tok_acc': tok_acc,
        'span_f1': span_f1,
    }


def train_ate_model(model, train_loader, dev_loader, device,
                    lr=1e-3, epochs=30, patience=7, model_name="Model",
                    bio_tags_list=None):
    """
    Vòng lặp chính: Train ATE model với Early Stopping bằng Span F1.
    Trả về (best_model, history dict).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5)

    best_f1, best_state, wait = 0, None, 0
    history = {'train_loss': [], 'dev_loss': [], 'dev_tok_acc': [], 'dev_span_f1': [], 'lr': []}

    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"{'='*60}")

    for ep in range(epochs):
        t0 = time.time()
        train_loss = train_epoch_ate(model, train_loader, optimizer, device)
        dev_res = predict_ate(model, dev_loader, device, bio_tags_list=bio_tags_list)

        tok_acc = dev_res['tok_acc']
        span_f1 = dev_res['span_f1']

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(span_f1)

        history['train_loss'].append(train_loss)
        history['dev_loss'].append(dev_res['loss'])
        history['dev_tok_acc'].append(tok_acc)
        history['dev_span_f1'].append(span_f1)
        history['lr'].append(current_lr)

        mark = ''
        if span_f1 > best_f1:
            best_f1 = span_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            mark = ' ***'
        else:
            wait += 1

        elapsed = time.time() - t0
        print(f"  Ep {ep+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Dev Loss: {dev_res['loss']:.4f} | TokAcc: {tok_acc:.4f} | "
              f"SpanF1: {span_f1:.4f} | LR: {current_lr:.6f} | {elapsed:.1f}s{mark}")

        if wait >= patience:
            print(f"  Early stopping at epoch {ep+1} (Best Span F1: {best_f1:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    return model, history


# ============================================================
# 2. ASC TRAINING ENGINE (Classification)
# ============================================================
def train_epoch_asc(model, loader, optimizer, criterion, device):
    """Một epoch huấn luyện cho ASC (CrossEntropy)"""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(loader, leave=False, desc="Training"):
        optimizer.zero_grad()
        seqs = batch['seq'].to(device)
        labels = batch['label'].to(device)

        logits = model(seqs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def predict_asc(model, loader, criterion, device):
    """Dự đoán cho ASC"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            seqs = batch['seq'].to(device)
            labels = batch['label'].to(device)

            logits = model(seqs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    from sklearn.metrics import f1_score
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return {
        'loss': total_loss / len(loader),
        'preds': all_preds,
        'labels': all_labels,
        'accuracy': acc,
        'macro_f1': macro_f1
    }


def train_asc_model(model, train_loader, dev_loader, device,
                    lr=1e-3, epochs=30, patience=7, model_name="Model",
                    class_weights=None):
    """Vòng lặp chính: Train ASC model với optional class-weighted loss"""
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5)

    best_f1, best_state, wait = 0, None, 0
    history = {'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_f1': [], 'lr': []}

    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"{'='*60}")

    for ep in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch_asc(model, train_loader, optimizer, criterion, device)
        dev_res = predict_asc(model, dev_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(dev_res['macro_f1'])

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['dev_loss'].append(dev_res['loss'])
        history['dev_f1'].append(dev_res['macro_f1'])
        history['lr'].append(current_lr)

        mark = ''
        if dev_res['macro_f1'] > best_f1:
            best_f1 = dev_res['macro_f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            mark = ' ***'
        else:
            wait += 1

        elapsed = time.time() - t0
        print(f"  Ep {ep+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Dev Loss: {dev_res['loss']:.4f} | Dev Mac_F1: {dev_res['macro_f1']:.4f} | "
              f"LR: {current_lr:.6f} | {elapsed:.1f}s{mark}")

        if wait >= patience:
            print(f"  Early stopping at epoch {ep+1} (Best Macro F1: {best_f1:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    return model, history


# ============================================================
# 3. E2E (PhoBERT) TRAINING ENGINE
# ============================================================
def train_epoch_e2e(model, loader, optimizer, scheduler, device, grad_accum=2):
    """Một epoch huấn luyện cho E2E PhoBERT-CRF với Word-Level Alignment + Gradient Accumulation"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, leave=False, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        word_ids = batch['word_ids'].to(device)
        word_count = batch['word_count'].to(device)
        tags = batch['word_tags'].to(device)

        loss = model(input_ids, attention_mask, word_ids, word_count, labels=tags)
        loss = loss / grad_accum
        loss.backward()
        total_loss += loss.item() * grad_accum

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Handle remaining gradients
    if (step + 1) % grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def predict_e2e(model, loader, device):
    """Dự đoán cho E2E và tính Span F1"""
    import torch
    from src.e2e.e2e_dataset import BIO_TAGS
    from src.utils.metrics import bio_tags_to_spans, evaluate_spans_f1

    model.eval()
    all_pred_tags, all_true_tags, all_lengths = [], [], []
    total_loss = 0
    correct, total_tokens = 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            word_ids = batch['word_ids'].to(device)
            word_count = batch['word_count'].to(device)
            tags = batch['word_tags'].to(device)

            loss = model(input_ids, attention_mask, word_ids, word_count, labels=tags)
            total_loss += loss.item()

            pred_tags = model(input_ids, attention_mask, word_ids, word_count)

            for i in range(len(word_count)):
                length = word_count[i].item()
                if length == 0: continue
                pred = pred_tags[i][:length]
                true = tags[i][:length].cpu().tolist()
                
                all_pred_tags.append(pred)
                all_true_tags.append(true)
                all_lengths.append(length)
                
                for p, t in zip(pred, true):
                    if p == t: correct += 1
                    total_tokens += 1

    tok_acc = correct / max(1, total_tokens)
    
    # Calculate Span F1
    ps = [bio_tags_to_spans(pt, BIO_TAGS, l) for pt, l in zip(all_pred_tags, all_lengths)]
    ts = [bio_tags_to_spans(tt, BIO_TAGS, l) for tt, l in zip(all_true_tags, all_lengths)]
    f1_res = evaluate_spans_f1(ps, ts)

    return {
        'loss': total_loss / len(loader),
        'pred_tags': all_pred_tags,
        'true_tags': all_true_tags,
        'lengths': all_lengths,
        'tok_acc': tok_acc,
        'span_f1': f1_res['f1']
    }


def train_e2e_model(model, train_loader, dev_loader, device,
                    lr=2e-5, epochs=15, patience=5, grad_accum=2, model_name="PhoBERT-CRF"):
    """Vòng lặp chính: Train E2E PhoBERT model với Differential LR, Gradient Accumulation và Early Stopping bằng Span F1"""
    import time
    import torch
    from transformers import get_linear_schedule_with_warmup

    bert_params = list(model.phobert.parameters())
    head_params = list(model.hidden2tag.parameters()) + list(model.crf.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': lr, 'weight_decay': 0.01},
        {'params': head_params, 'lr': 1e-3, 'weight_decay': 0.0},
    ])
    
    total_steps = len(train_loader) * epochs // grad_accum
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_metric, best_state, wait = 0, None, 0
    history = {'train_loss': [], 'dev_loss': [], 'dev_span_f1': [], 'lr': []}

    batch_size = train_loader.batch_size or 16
    print(f"\n{'='*60}")
    print(f"  Training {model_name} (PhoBERT) - Differential LR")
    print(f"  BERT LR: {lr}, Head LR: 1e-3")
    print(f"  Batch: {batch_size} x {grad_accum} accum = {batch_size * grad_accum} effective")
    print(f"  Total steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"{'='*60}")

    for ep in range(epochs):
        t0 = time.time()
        train_loss = train_epoch_e2e(model, train_loader, optimizer, scheduler, device, grad_accum=grad_accum)
        dev_res = predict_e2e(model, dev_loader, device)

        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['dev_loss'].append(dev_res['loss'])
        history['dev_span_f1'].append(dev_res['span_f1'])
        history['lr'].append(current_lr)

        mark = ''
        if dev_res['span_f1'] > best_metric:
            best_metric = dev_res['span_f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            mark = ' ***'
        else:
            wait += 1

        elapsed = time.time() - t0
        print(f"  Ep {ep+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Dev Loss: {dev_res['loss']:.4f} | TokAcc: {dev_res['tok_acc']:.4f} | "
              f"SpanF1: {dev_res['span_f1']:.4f} | LR: {current_lr:.2e} | {elapsed:.1f}s{mark}")

        if wait >= patience:
            print(f"  Early stopping at epoch {ep+1} (Best Span F1: {best_metric:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    return model, history
