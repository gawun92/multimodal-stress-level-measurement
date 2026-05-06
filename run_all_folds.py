import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
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

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()





def train_fold(fold, label="binary-stress", epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE,
               lr=config.LEARNING_RATE, windowed=False, max_frames=config.MAX_FRAMES,
               seed=config.RANDOM_SEED):
    num_classes = config.NUM_CLASSES_BINARY if label == "binary-stress" else config.NUM_CLASSES_AFFECT3

    train_subjects, val_subjects, test_subjects = get_subject_splits(fold=fold)
    train_ds = StressAudioDataset(train_subjects, label_col=label, augment=True, windowed=windowed)
    val_ds = StressAudioDataset(val_subjects, label_col=label, augment=False, windowed=windowed)

    logging.info(f"\n{'='*60}")
    logging.info(f"FOLD {fold}: train={len(train_ds)} | val={len(val_ds)} samples")
    logging.info(f"  Train subjects: {len(train_subjects)} | Val: {len(val_subjects)} | Test: {len(test_subjects)}")
    logging.info(f"{'='*60}")

    if len(train_ds) == 0:
        logging.info(f"  [SKIP] No training samples for fold {fold}")
        return None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    torch.manual_seed(seed)
    np.random.seed(seed)

    branch = AudioBranch(
        n_mels=config.N_MELS, max_frames=max_frames,
        cnn_channels=config.CNN_CHANNELS, embed_dim=config.EMBED_DIM,
        n_heads=config.TRANSFORMER_HEADS, n_layers=config.TRANSFORMER_LAYERS,
        ff_dim=config.TRANSFORMER_FF_DIM, dropout=config.TRANSFORMER_DROPOUT,
    )
    model = AudioClassifier(branch, num_classes=num_classes).to(DEVICE)

    labels_arr = np.array([lbl for _, lbl in train_ds.samples])
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels_arr)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    logging.info(f"  Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1_macro": []}
    best_val_f1 = -1.0
    patience_counter = 0

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"audio_branch_fold{fold}_seed{seed}_{label}.pt")

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for X, y in train_pbar:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_loss_total, val_batches = 0.0, 0
        val_preds_all, val_labels_all = [], []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for X, y in val_pbar:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = model(X)
                val_loss_total += criterion(logits, y).item()
                val_preds_all.append(logits.argmax(dim=1).cpu().numpy())
                val_labels_all.append(y.cpu().numpy())
                val_batches += 1
        val_loss = val_loss_total / max(val_batches, 1)
        val_preds_np = np.concatenate(val_preds_all)
        val_labels_np = np.concatenate(val_labels_all)
        val_acc = (val_preds_np == val_labels_np).mean()
        val_f1_macro = f1_score(val_labels_np, val_preds_np, average="macro", zero_division=0)

        scheduler.step(val_f1_macro)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_f1_macro)

        lr_now = optimizer.param_groups[0]["lr"]
        logging.info(f"  Epoch {epoch+1:3d}/{epochs} | train_loss={train_loss:.4f} | "
                     f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
                     f"val_f1m={val_f1_macro:.4f} | lr={lr_now:.2e}")

        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logging.info(f"  Early stopping at epoch {epoch+1}")
                break

    logging.info(f"  Best val macro F1: {best_val_f1:.4f} | Saved: {ckpt_path}")
    return history


def _load_model(fold, seed, label, num_classes):
    branch = AudioBranch(
        n_mels=config.N_MELS, max_frames=config.MAX_FRAMES,
        cnn_channels=config.CNN_CHANNELS, embed_dim=config.EMBED_DIM,
        n_heads=config.TRANSFORMER_HEADS, n_layers=config.TRANSFORMER_LAYERS,
        ff_dim=config.TRANSFORMER_FF_DIM, dropout=config.TRANSFORMER_DROPOUT,
    )
    model = AudioClassifier(branch, num_classes=num_classes)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"audio_branch_fold{fold}_seed{seed}_{label}.pt")
    if not os.path.exists(ckpt_path):
        logging.info(f"  [WARN] Checkpoint not found: {ckpt_path}")
        return None
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


def _ensemble_tta_probs(models, X, tta_steps=1):
    all_probs = []
    with torch.no_grad():
        for model in models:
            logits = model(X)
            all_probs.append(torch.softmax(logits, dim=1).cpu())
    return torch.stack(all_probs).mean(dim=0).numpy()


def tune_fold_threshold(fold, label, seeds, tta_steps, val_subjects):
    if label != "binary-stress":
        return
    num_classes = config.NUM_CLASSES_BINARY

    models = []
    for s in seeds:
        m = _load_model(fold, s, label, num_classes)
        if m is not None:
            models.append(m)
    if not models:
        logging.info(f"  [SKIP] No checkpoints for fold {fold} threshold tuning")
        return

    val_ds = StressAudioDataset(val_subjects, label_col=label, augment=False)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)

    val_probs_all, val_labels_all = [], []
    for X, y in val_loader:
        probs = _ensemble_tta_probs(models, X.to(DEVICE), tta_steps)
        val_probs_all.append(probs)
        val_labels_all.append(y.numpy())
    val_probs_np = np.concatenate(val_probs_all)[:, 1]
    val_labels_np = np.concatenate(val_labels_all)

    best_thresh, best_thresh_f1 = 0.5, 0.0
    MIN_CLASS_RECALL = 0.10
    for t in np.arange(0.05, 0.96, 0.01):
        preds_t = (val_probs_np >= t).astype(int)
        recall_0 = ((preds_t == 0) & (val_labels_np == 0)).sum() / max((val_labels_np == 0).sum(), 1)
        recall_1 = ((preds_t == 1) & (val_labels_np == 1)).sum() / max((val_labels_np == 1).sum(), 1)
        if recall_0 < MIN_CLASS_RECALL or recall_1 < MIN_CLASS_RECALL:
            continue
        f1_t = f1_score(val_labels_np, preds_t, average="macro", zero_division=0)
        if f1_t > best_thresh_f1:
            best_thresh_f1, best_thresh = f1_t, float(t)

    thresh_path = os.path.join(config.CHECKPOINT_DIR, f"audio_branch_fold{fold}_{label}_threshold.json")
    with open(thresh_path, "w") as f:
        json.dump({"threshold": best_thresh, "val_macro_f1": best_thresh_f1}, f)
    logging.info(f"  Ensemble threshold fold {fold}: {best_thresh:.2f} (val F1m: {best_thresh_f1:.4f})")


def evaluate_fold(fold, label="binary-stress", windowed=False, max_frames=config.MAX_FRAMES,
                  seeds=None, tta_steps=config.TTA_STEPS):
    import pandas as pd
    from pathlib import Path as _Path

    if seeds is None:
        seeds = config.ENSEMBLE_SEEDS
    num_classes = config.NUM_CLASSES_BINARY if label == "binary-stress" else config.NUM_CLASSES_AFFECT3

    _, _, test_subjects = get_subject_splits(fold=fold)

    models = []
    for s in seeds:
        m = _load_model(fold, s, label, num_classes)
        if m is not None:
            models.append(m)
    if not models:
        logging.info(f"  [SKIP] No checkpoints found for fold {fold}")
        return None
    logging.info(f"  ensemble: {len(models)} models loaded for fold {fold}")

    if windowed:
        labels_df = pd.read_csv(config.LABELS_CSV).set_index("subject/task")
        mel_dir = config.MEL_WINDOWED_DIR

        clip_preds, clip_labels, clip_probs = [], [], []
        for subject_id in test_subjects:
            for task in config.AUDIO_TASKS:
                key = f"{subject_id}_{task}"
                if key not in labels_df.index:
                    continue
                window_files = sorted(_Path(mel_dir).glob(f"{subject_id}/{task}_mel_w*.npy"))
                if not window_files:
                    continue
                clip_label = int(labels_df.loc[key, label])
                windows = np.stack([np.load(str(f)) for f in window_files])
                X = torch.from_numpy(windows).float().to(DEVICE)
                prob = _ensemble_tta_probs(models, X, tta_steps).mean(axis=0)
                clip_probs.append(prob)
                clip_labels.append(clip_label)
        preds = None
        labels = np.array(clip_labels)
        probs = np.array(clip_probs)
        logging.info(f"\n--- Fold {fold} Test [CLIP-LEVEL, {len(probs)} clips, {len(test_subjects)} subjects] ---")

    else:
        test_ds = StressAudioDataset(test_subjects, label_col=label, windowed=False)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=False)
        all_labels_l, all_probs_l = [], []
        for X, y in test_loader:
            probs_b = _ensemble_tta_probs(models, X.to(DEVICE), tta_steps)
            all_labels_l.append(y.numpy())
            all_probs_l.append(probs_b)
        labels = np.concatenate(all_labels_l)
        probs = np.concatenate(all_probs_l)
        logging.info(f"\n--- Fold {fold} Test Results ({len(test_ds)} samples, {len(test_subjects)} subjects) ---")

    threshold = 0.5
    if num_classes == 2:
        thresh_path = os.path.join(
            config.CHECKPOINT_DIR, f"audio_branch_fold{fold}_{label}_threshold.json"
        )
        if os.path.exists(thresh_path):
            with open(thresh_path) as f:
                threshold = json.load(f)["threshold"]
            logging.info(f"  Using tuned threshold: {threshold:.2f}  (loaded from {thresh_path})")
        else:
            logging.info(f"  Using default threshold: 0.50  (no tuned threshold found)")
        preds = (probs[:, 1] >= threshold).astype(int)
    else:
        preds = probs.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)

    if num_classes == 2:
        auc_roc = roc_auc_score(labels, probs[:, 1])
    else:
        auc_roc = roc_auc_score(labels, probs, multi_class="ovr")

    target_names = ["no-stress", "stressed"] if label == "binary-stress" else ["class-0", "class-1", "class-2"]
    report = classification_report(labels, preds, target_names=target_names, zero_division=0)

    logging.info(f"  Accuracy:       {acc:.4f}")
    logging.info(f"  Weighted F1:    {f1_w:.4f}")
    logging.info(f"  Macro F1:       {f1_macro:.4f}")
    logging.info(f"  AUC-ROC:        {auc_roc:.4f}")
    logging.info(f"  Balanced Acc:   {bal_acc:.4f}")
    logging.info(f"  MCC:            {mcc:.4f}")
    logging.info(f"\n{report}")

    return {
        "fold": fold,
        "n_test_subjects": len(test_subjects),
        "n_test_samples": len(preds),
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


def evaluate_held_out(best_fold, label="binary-stress", seeds=None):
    if seeds is None:
        seeds = config.ENSEMBLE_SEEDS

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

    models = []
    for s in seeds:
        m = _load_model(best_fold, s, label, num_classes)
        if m is not None:
            models.append(m)
    if not models:
        logging.info(f"  [SKIP] No seed checkpoints found for fold {best_fold}.")
        return None

    all_labels, all_probs = [], []
    for X, y in test_loader:
        probs_b = _ensemble_tta_probs(models, X.to(DEVICE))
        all_labels.append(y.numpy())
        all_probs.append(probs_b)

    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    preds = probs.argmax(axis=1)

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
    logging.info(f"  Using ensemble from best CV fold: fold {best_fold} ({len(models)} seeds)")
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


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="binary-stress",
                   choices=["binary-stress", "affect3-class"])
    p.add_argument("--windowed", action="store_true",
                   help="Use sliding-window mels (~8x more samples per clip)")
    return p.parse_args()


def main():
    args = parse_args()
    label = args.label
    windowed = args.windowed
    max_frames = config.WINDOW_FRAMES if windowed else config.MAX_FRAMES
    logging.info(f"Mode: {'windowed (' + str(config.WINDOW_SEC) + 's window, ' + str(config.HOP_SEC) + 's hop)' if windowed else 'full-clip'} | max_frames={max_frames}")
    n_folds = config.NUM_FOLDS

    seeds = config.ENSEMBLE_SEEDS
    tta_steps = config.TTA_STEPS
    logging.info(f"Ensemble: {len(seeds)} seeds {seeds} | TTA steps: {tta_steps}")

    all_histories = {}
    for fold in range(n_folds):
        _, val_subjects, _ = get_subject_splits(fold=fold)
        for seed in seeds:
            logging.info(f"\n>>> Fold {fold} | Seed {seed}")
            history = train_fold(fold, label=label, windowed=windowed,
                                 max_frames=max_frames, seed=seed)
            if history:
                all_histories[(fold, seed)] = history
        tune_fold_threshold(fold, label, seeds, tta_steps, val_subjects)

    all_results = []
    for fold in range(n_folds):
        result = evaluate_fold(fold, label=label, windowed=windowed, max_frames=max_frames,
                               seeds=seeds, tta_steps=tta_steps)
        if result:
            all_results.append(result)

    accs = [r["accuracy"] for r in all_results]
    f1s = [r["f1_weighted"] for r in all_results]
    f1_macros = [r["f1_macro"] for r in all_results]
    aucs = [r["auc_roc"] for r in all_results]
    bal_accs = [r["balanced_accuracy"] for r in all_results]
    mccs = [r["mcc"] for r in all_results]

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
    logging.info(f"  Mean Accuracy:    {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    logging.info(f"  Mean Weighted F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    logging.info(f"  Mean Macro F1:    {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
    logging.info(f"  Mean AUC-ROC:     {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    logging.info(f"  Mean Balanced:    {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f}")
    logging.info(f"  Mean MCC:         {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")
    logging.info(f"{'='*70}")

    best_fold = int(np.argmax(f1_macros))
    logging.info(
        f"\nBest CV fold by Macro F1: fold {best_fold} "
        f"(F1m={f1_macros[best_fold]:.4f} | acc={accs[best_fold]:.4f} | "
        f"AUC={aucs[best_fold]:.4f})"
    )
    held_out_result = evaluate_held_out(best_fold, label=label, seeds=seeds)

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
