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
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def predict_ate(model, loader, device):
    """Dự đoán toàn bộ tập dữ liệu cho ATE. Trả về pred_tags, true_tags, lengths"""
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

    return {
        'loss': total_loss / len(loader),
        'pred_tags': all_pred_tags,
        'true_tags': all_true_tags,
        'lengths': all_lengths
    }


def train_ate_model(model, train_loader, dev_loader, device,
                    lr=1e-3, epochs=30, patience=7, model_name="Model"):
    """
    Vòng lặp chính: Train ATE model với Early Stopping.
    Trả về (best_model, history dict).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_f1, best_state, wait = 0, None, 0
    history = {'train_loss': [], 'dev_loss': [], 'dev_tok_acc': [], 'lr': []}

    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"{'='*60}")

    for ep in range(epochs):
        t0 = time.time()
        train_loss = train_epoch_ate(model, train_loader, optimizer, device)
        dev_res = predict_ate(model, dev_loader, device)

        # Token accuracy
        correct, total = 0, 0
        for pt, tt in zip(dev_res['pred_tags'], dev_res['true_tags']):
            for p, t in zip(pt, tt):
                if p == t: correct += 1
                total += 1
        tok_acc = correct / max(1, total)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(dev_res['loss'])

        history['train_loss'].append(train_loss)
        history['dev_loss'].append(dev_res['loss'])
        history['dev_tok_acc'].append(tok_acc)
        history['lr'].append(current_lr)

        mark = ''
        if tok_acc > best_f1:
            best_f1 = tok_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            mark = ' ***'
        else:
            wait += 1

        elapsed = time.time() - t0
        print(f"  Ep {ep+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Dev Loss: {dev_res['loss']:.4f} | TokAcc: {tok_acc:.4f} | "
              f"LR: {current_lr:.6f} | {elapsed:.1f}s{mark}")

        if wait >= patience:
            print(f"  Early stopping at epoch {ep+1}")
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
                    lr=1e-3, epochs=30, patience=7, model_name="Model"):
    """Vòng lặp chính: Train ASC model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

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
        scheduler.step(dev_res['loss'])

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
def train_epoch_e2e(model, loader, optimizer, scheduler, device):
    """Một epoch huấn luyện cho E2E PhoBERT-CRF với Word-Level Alignment"""
    model.train()
    total_loss = 0
    for batch in tqdm(loader, leave=False, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        word_ids = batch['word_ids'].to(device)
        word_count = batch['word_count'].to(device)
        tags = batch['word_tags'].to(device)

        loss = model(input_ids, attention_mask, word_ids, word_count, labels=tags)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def predict_e2e(model, loader, device):
    """Dự đoán cho E2E. Căn cứ trên word_count thực tế để cắt tags."""
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

            # Loss
            loss = model(input_ids, attention_mask, word_ids, word_count, labels=tags)
            total_loss += loss.item()

            # Decode
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

    return {
        'loss': total_loss / len(loader),
        'pred_tags': all_pred_tags,
        'true_tags': all_true_tags,
        'lengths': all_lengths,
        'tok_acc': tok_acc
    }


def train_e2e_model(model, train_loader, dev_loader, device,
                    lr=2e-5, epochs=15, patience=5, model_name="PhoBERT-CRF"):
    """Vòng lặp chính: Train E2E PhoBERT model với Differential LR và Early Stopping bằng Acc/F1"""
    from transformers import get_linear_schedule_with_warmup

    # Differential Learning Rates
    bert_params = list(model.phobert.parameters())
    head_params = list(model.hidden2tag.parameters()) + list(model.crf.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': lr, 'weight_decay': 0.01},
        {'params': head_params, 'lr': 1e-3, 'weight_decay': 0.0},
    ])
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_metric, best_state, wait = 0, None, 0
    history = {'train_loss': [], 'dev_loss': [], 'dev_tok_acc': [], 'lr': []}

    print(f"\n{'='*60}")
    print(f"  Training {model_name} (PhoBERT) - Differential LR")
    print(f"{'='*60}")

    for ep in range(epochs):
        t0 = time.time()
        train_loss = train_epoch_e2e(model, train_loader, optimizer, scheduler, device)
        dev_res = predict_e2e(model, dev_loader, device)

        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['dev_loss'].append(dev_res['loss'])
        history['dev_tok_acc'].append(dev_res['tok_acc'])
        history['lr'].append(current_lr)

        mark = ''
        if dev_res['tok_acc'] > best_metric:
            best_metric = dev_res['tok_acc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            mark = ' ***'
        else:
            wait += 1

        elapsed = time.time() - t0
        print(f"  Ep {ep+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Dev Loss: {dev_res['loss']:.4f} | TokAcc: {dev_res['tok_acc']:.4f} | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s{mark}")

        if wait >= patience:
            print(f"  Early stopping at epoch {ep+1} (Best TokAcc: {best_metric:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    return model, history
