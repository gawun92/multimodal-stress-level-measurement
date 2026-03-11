"""
run_all_folds.py

Runs training + evaluation for all 5 folds sequentially,
captures results, and generates comprehensive analysis graphics.

Usage:
    python run_all_folds.py
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_score, recall_score, roc_auc_score,
    matthews_corrcoef, balanced_accuracy_score,
)
import torch.nn.functional as F
from tqdm import tqdm

import config
from dataset import StressAudioDataset, get_subject_splits, get_held_out_subjects
from models.audio_branch import AudioBranch, AudioClassifier

# Use MPS for speed, with pin_memory=False to avoid DataLoader hangs
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")





def train_fold(fold, label="binary-stress", epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE):
    """Train a single fold and return training history."""
    num_classes = config.NUM_CLASSES_BINARY if label == "binary-stress" else config.NUM_CLASSES_AFFECT3

    train_subjects, val_subjects, test_subjects = get_subject_splits(fold=fold)
    train_ds = StressAudioDataset(train_subjects, label_col=label, augment=True)
    val_ds = StressAudioDataset(val_subjects, label_col=label, augment=False)

    logging.info(f"\n{'='*60}")
    logging.info(f"FOLD {fold}: train={len(train_ds)} | val={len(val_ds)} samples")
    logging.info(f"  Train subjects: {len(train_subjects)} | Val: {len(val_subjects)} | Test: {len(test_subjects)}")
    logging.info(f"{'='*60}")

    if len(train_ds) == 0:
        logging.info(f"  [SKIP] No training samples for fold {fold}")
        return None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Model
    branch = AudioBranch(
        n_mels=config.N_MELS, max_frames=config.MAX_FRAMES,
        cnn_channels=config.CNN_CHANNELS, embed_dim=config.EMBED_DIM,
        n_heads=config.TRANSFORMER_HEADS, n_layers=config.TRANSFORMER_LAYERS,
        ff_dim=config.TRANSFORMER_FF_DIM, dropout=config.TRANSFORMER_DROPOUT,
    )
    model = AudioClassifier(branch, num_classes=num_classes).to(DEVICE)

    # Class weights
    labels_arr = np.array([lbl for _, lbl in train_ds.samples])
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels_arr)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    # Aggressively weight the underrepresented class
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"audio_branch_fold{fold}_{label}.pt")

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss, n_batches = 0.0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for X, y in train_pbar:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss_total, correct, total, val_batches = 0.0, 0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for X, y in val_pbar:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = model(X)
                loss = criterion(logits, y)
                val_loss_total += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                val_batches += 1
        val_loss = val_loss_total / max(val_batches, 1)
        val_acc = correct / max(total, 1)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        logging.info(f"  Epoch {epoch+1:3d}/{epochs} | train_loss={train_loss:.4f} | "
                     f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | lr={lr_now:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logging.info(f"  Early stopping at epoch {epoch+1}")
                break

    logging.info(f"  Best val_loss: {best_val_loss:.4f} | Saved: {ckpt_path}")
    return history


def evaluate_fold(fold, label="binary-stress"):
    """Evaluate a single fold and return metrics."""
    num_classes = config.NUM_CLASSES_BINARY if label == "binary-stress" else config.NUM_CLASSES_AFFECT3

    _, _, test_subjects = get_subject_splits(fold=fold)
    test_ds = StressAudioDataset(test_subjects, label_col=label)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    branch = AudioBranch(
        n_mels=config.N_MELS, max_frames=config.MAX_FRAMES,
        cnn_channels=config.CNN_CHANNELS, embed_dim=config.EMBED_DIM,
        n_heads=config.TRANSFORMER_HEADS, n_layers=config.TRANSFORMER_LAYERS,
        ff_dim=config.TRANSFORMER_FF_DIM, dropout=config.TRANSFORMER_DROPOUT,
    )
    model = AudioClassifier(branch, num_classes=num_classes)

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"audio_branch_fold{fold}_{label}.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(labels, preds)
    mcc = matthews_corrcoef(labels, preds)          # gold std for imbalanced binary
    bal_acc = balanced_accuracy_score(labels, preds) # = (sensitivity + specificity) / 2

    # AUC-ROC
    if num_classes == 2:
        auc_roc = roc_auc_score(labels, probs[:, 1])
    else:
        auc_roc = roc_auc_score(labels, probs, multi_class="ovr")

    target_names = ["no-stress", "stressed"] if label == "binary-stress" else ["class-0", "class-1", "class-2"]
    report = classification_report(labels, preds, target_names=target_names, zero_division=0)

    logging.info(f"\n--- Fold {fold} Test Results ({len(test_ds)} samples, {len(test_subjects)} subjects) ---")
    logging.info(f"  Accuracy:       {acc:.4f}")
    logging.info(f"  Weighted F1:    {f1_w:.4f}")
    logging.info(f"  Macro F1:       {f1_macro:.4f}")
    logging.info(f"  AUC-ROC:        {auc_roc:.4f}")
    logging.info(f"  Balanced Acc:   {bal_acc:.4f}  (0.5 = random, detects collapse)")
    logging.info(f"  MCC:            {mcc:.4f}  (0 = random, -1/+1 = worst/best)")
    logging.info(f"\n{report}")

    return {
        "fold": fold,
        "n_test_subjects": len(test_subjects),
        "n_test_samples": len(test_ds),
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_macro,
        "auc_roc": auc_roc,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm.tolist(),
        "preds": preds.tolist(),
        "labels": labels.tolist(),
        "probs": probs.tolist(),
    }


def evaluate_held_out(best_fold, label="binary-stress"):
    """
    Evaluate on the held-out subjects (never seen during training/CV)
    using the checkpoint from the best CV fold.
    """
    held_out = get_held_out_subjects()
    if not held_out:
        logging.info("  [SKIP] No held-out subjects with mel spectrograms found.")
        return None

    num_classes = config.NUM_CLASSES_BINARY if label == "binary-stress" else config.NUM_CLASSES_AFFECT3
    test_ds = StressAudioDataset(held_out, label_col=label, augment=False)
    if len(test_ds) == 0:
        logging.info("  [SKIP] Held-out subjects have no labeled samples.")
        return None

    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=False)

    branch = AudioBranch(
        n_mels=config.N_MELS, max_frames=config.MAX_FRAMES,
        cnn_channels=config.CNN_CHANNELS, embed_dim=config.EMBED_DIM,
        n_heads=config.TRANSFORMER_HEADS, n_layers=config.TRANSFORMER_LAYERS,
        ff_dim=config.TRANSFORMER_FF_DIM, dropout=config.TRANSFORMER_DROPOUT,
    )
    model = AudioClassifier(branch, num_classes=num_classes)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"audio_branch_fold{best_fold}_{label}.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    if num_classes == 2:
        auc_roc = roc_auc_score(labels, probs[:, 1])
    else:
        auc_roc = roc_auc_score(labels, probs, multi_class="ovr")

    target_names = ["no-stress", "stressed"] if label == "binary-stress" else ["class-0", "class-1", "class-2"]
    report = classification_report(labels, preds, target_names=target_names, zero_division=0)
    cm = confusion_matrix(labels, preds)

    logging.info(f"\n{'='*60}")
    logging.info(f"HELD-OUT EVALUATION ({len(held_out)} subjects: {held_out})")
    logging.info(f"  Using checkpoint from best CV fold: fold {best_fold}")
    logging.info(f"  Samples: {len(test_ds)}")
    logging.info(f"{'='*60}")
    logging.info(f"  Accuracy:     {acc:.4f}")
    logging.info(f"  Weighted F1:  {f1_w:.4f}")
    logging.info(f"  Macro F1:     {f1_macro:.4f}")
    logging.info(f"  AUC-ROC:      {auc_roc:.4f}")
    logging.info(f"  Balanced Acc: {bal_acc:.4f}")
    logging.info(f"  MCC:          {mcc:.4f}")
    logging.info(f"\n{report}")
    logging.info(f"  Confusion Matrix:\n{cm}")

    return {
        "subjects": held_out,
        "best_fold_used": best_fold,
        "n_samples": len(test_ds),
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_macro,
        "auc_roc": auc_roc,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "confusion_matrix": cm.tolist(),
        "preds": preds.tolist(),
        "labels": labels.tolist(),
        "probs": probs.tolist(),
    }


def main():
    label = "binary-stress"
    n_folds = config.NUM_FOLDS

    # ── Train all folds ──
    all_histories = {}
    for fold in range(n_folds):
        history = train_fold(fold, label=label)
        if history:
            all_histories[fold] = history

    # ── Evaluate all folds ──
    all_results = []
    for fold in range(n_folds):
        result = evaluate_fold(fold, label=label)
        all_results.append(result)

    # ── Print CV Summary ──
    accs      = [r["accuracy"]          for r in all_results]
    f1s       = [r["f1_weighted"]       for r in all_results]
    f1_macros = [r["f1_macro"]          for r in all_results]
    aucs      = [r["auc_roc"]           for r in all_results]
    bal_accs  = [r["balanced_accuracy"] for r in all_results]
    mccs      = [r["mcc"]               for r in all_results]

    logging.info(f"\n{'='*70}")
    logging.info(f"5-FOLD CROSS-VALIDATION SUMMARY")
    logging.info(f"{'='*70}")
    for r in all_results:
        logging.info(
            f"  Fold {r['fold']}: acc={r['accuracy']:.4f} | F1w={r['f1_weighted']:.4f} | "
            f"F1m={r['f1_macro']:.4f} | AUC={r['auc_roc']:.4f} | "
            f"BalAcc={r['balanced_accuracy']:.4f} | MCC={r['mcc']:.4f}"
        )
    logging.info(f"{'='*70}")
    logging.info(f"  Mean Accuracy:     {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    logging.info(f"  Mean Weighted F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    logging.info(f"  Mean Macro F1:     {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
    logging.info(f"  Mean AUC-ROC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    logging.info(f"  Mean Balanced Acc: {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f}")
    logging.info(f"  Mean MCC:          {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")
    logging.info(f"{'='*70}")

    # ── Held-Out Evaluation ──
    # Select best fold by Macro F1 (most robust for imbalanced binary —
    # not fooled by "all-stressed" collapse which inflates raw accuracy).
    best_fold = int(np.argmax(f1_macros))
    logging.info(
        f"\nBest CV fold by Macro F1: fold {best_fold} "
        f"(F1m={f1_macros[best_fold]:.4f} | acc={accs[best_fold]:.4f} | "
        f"AUC={aucs[best_fold]:.4f})"
    )
    held_out_result = evaluate_held_out(best_fold, label=label)

    # ── Save results ──
    results_path = os.path.join(config.BASE_DIR, "cv_results.json")
    save_data = {
        "label": label,
        "n_folds": n_folds,
        "held_out_subjects": config.HELD_OUT_SUBJECTS,
        "fold_results": all_results,
        "held_out_result": held_out_result,
        "histories": {str(k): v for k, v in all_histories.items()},
    }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    # Setup dual logging (Console + File)
    log_path = os.path.join(config.BASE_DIR, "cv_results.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Starting Cross-Validation... Logs will be saved to {log_path}\n")
    main()
