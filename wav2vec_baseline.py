"""
wav2vec_baseline.py

Wav2Vec2-based audio baseline for affect3-class stress classification.
Uses IDENTICAL subject-level 5-fold CV splits as our mel-spectrogram pipeline
so the comparison is apples-to-apples.

Architecture:
    facebook/wav2vec2-base (frozen) -> mean-pool over time -> 768-d embedding
    -> MLP head (768 -> 256 -> ReLU -> Dropout(0.3) -> 3)

The backbone stays frozen (feature extraction mode) — same pattern as the
StressID baseline repo (mean/std aggregation + classifier on W2V features).

Usage:
    python wav2vec_baseline.py                          # affect3-class (default)
    python wav2vec_baseline.py --label binary-stress
    python wav2vec_baseline.py --max-seconds 15         # shorter clip for speed
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2Model

import config
from dataset import get_subject_splits, get_held_out_subjects

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
SAMPLE_RATE = 16_000          # wav2vec2 expects 16 kHz
DEFAULT_MAX_SECONDS = 30      # truncate/pad audio to this length
AUDIO_DIR = os.path.join(config.DATA_DIR, "Audio")
W2V_EMBED_DIM = 768           # facebook/wav2vec2-base hidden size
W2V_MODEL_ID = "facebook/wav2vec2-base"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────

class Wav2VecDataset(Dataset):
    """
    Loads raw waveforms from Audio/{subject_id}/{subject_id}_{task}.wav.
    Resamples to 16 kHz, truncates/zero-pads to max_samples.
    Returns (waveform: FloatTensor [max_samples], label: LongTensor).
    """

    def __init__(self, subject_ids, label_col="affect3-class",
                 max_seconds=DEFAULT_MAX_SECONDS):
        import pandas as pd

        self.max_samples = max_seconds * SAMPLE_RATE
        self.label_col = label_col

        labels_df = pd.read_csv(config.LABELS_CSV).set_index("subject/task")

        self.samples = []
        for subject_id in subject_ids:
            for task in config.AUDIO_TASKS:
                key = f"{subject_id}_{task}"
                wav_path = os.path.join(
                    AUDIO_DIR, subject_id, f"{subject_id}_{task}.wav"
                )

                if key not in labels_df.index:
                    continue
                if not os.path.exists(wav_path):
                    continue

                label = int(labels_df.loc[key, self.label_col])
                self.samples.append((wav_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]

        waveform, sr = torchaudio.load(wav_path)   # (channels, samples)
        waveform = waveform.mean(dim=0)             # mono: (samples,)

        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Truncate or zero-pad to fixed length
        n = waveform.size(0)
        if n >= self.max_samples:
            waveform = waveform[: self.max_samples]
        else:
            pad = torch.zeros(self.max_samples - n)
            waveform = torch.cat([waveform, pad])

        return waveform, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────

class Wav2VecClassifier(nn.Module):
    """
    Frozen Wav2Vec2 backbone + trainable MLP classification head.

    Backbone output: (batch, seq_len, 768)  ->  mean-pool  ->  (batch, 768)
    Head: Linear(768, 256) -> ReLU -> Dropout(0.3) -> Linear(256, num_classes)
    """

    def __init__(self, num_classes=3, hidden_dim=256):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(W2V_MODEL_ID)

        # Freeze the entire backbone — we use it purely as a feature extractor
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(W2V_EMBED_DIM, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, waveform):
        # waveform: (batch, max_samples)
        # Wav2Vec2 returns BaseModelOutput; .last_hidden_state is (batch, T, 768)
        with torch.no_grad():
            hidden = self.backbone(waveform).last_hidden_state  # (B, T, 768)
        embedding = hidden.mean(dim=1)                          # (B, 768)
        return self.head(embedding)                             # (B, num_classes)


# ─────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────

def get_class_weights(dataset, num_classes):
    labels = np.array([lbl for _, lbl in dataset.samples])
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, n = 0.0, 0
    for X, y in tqdm(loader, desc="  train", leave=False):
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total, n = 0.0, 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            total_loss += criterion(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            n += 1
    return total_loss / max(n, 1), correct / max(total, 1)


# ─────────────────────────────────────────
# Fold train + eval
# ─────────────────────────────────────────

def train_fold(fold, label, num_classes, max_seconds,
               epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE,
               lr=config.LEARNING_RATE):
    train_subjects, val_subjects, _ = get_subject_splits(fold=fold)

    train_ds = Wav2VecDataset(train_subjects, label_col=label, max_seconds=max_seconds)
    val_ds   = Wav2VecDataset(val_subjects,   label_col=label, max_seconds=max_seconds)

    logging.info(f"\n{'='*60}\nFOLD {fold}: train={len(train_ds)} | val={len(val_ds)} samples\n{'='*60}")
    if len(train_ds) == 0:
        logging.info(f"  [SKIP] No training samples for fold {fold}")
        return None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = Wav2VecClassifier(num_classes=num_classes).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"  Trainable params (head only): {trainable:,}")

    class_weights = get_class_weights(train_ds, num_classes)
    logging.info(f"  Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"wav2vec_fold{fold}_{label}.pt")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        logging.info(
            f"  Epoch {epoch+1:3d}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | lr={lr_now:.2e}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.head.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logging.info(f"  Early stopping at epoch {epoch+1}")
                break

    logging.info(f"  Best val_loss: {best_val_loss:.4f} | Saved: {ckpt_path}")
    return ckpt_path


def eval_fold(fold, label, num_classes, max_seconds):
    _, _, test_subjects = get_subject_splits(fold=fold)
    test_ds = Wav2VecDataset(test_subjects, label_col=label, max_seconds=max_seconds)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=False)

    model = Wav2VecClassifier(num_classes=num_classes).to(DEVICE)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"wav2vec_fold{fold}_{label}.pt")
    model.head.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs  = np.concatenate(all_probs)

    acc      = accuracy_score(labels, preds)
    f1_w     = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro",    zero_division=0)
    bal_acc  = balanced_accuracy_score(labels, preds)
    mcc      = matthews_corrcoef(labels, preds)
    auc_roc  = (roc_auc_score(labels, probs[:, 1])
                if num_classes == 2
                else roc_auc_score(labels, probs, multi_class="ovr"))

    target_names = (["no-stress", "stressed"] if label == "binary-stress"
                    else ["class-0", "class-1", "class-2"])
    report = classification_report(labels, preds, target_names=target_names, zero_division=0)
    cm     = confusion_matrix(labels, preds)

    logging.info(
        f"\n--- Fold {fold} Test ({len(test_ds)} samples, {len(test_subjects)} subjects) ---\n"
        f"  Accuracy:     {acc:.4f}\n"
        f"  Weighted F1:  {f1_w:.4f}\n"
        f"  Macro F1:     {f1_macro:.4f}\n"
        f"  AUC-ROC:      {auc_roc:.4f}\n"
        f"  Balanced Acc: {bal_acc:.4f}\n"
        f"  MCC:          {mcc:.4f}\n\n{report}"
    )
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
        "confusion_matrix": cm.tolist(),
        "preds": preds.tolist(),
        "labels": labels.tolist(),
        "probs": probs.tolist(),
    }


# ─────────────────────────────────────────
# Comparison printer
# ─────────────────────────────────────────

def print_comparison(w2v_results, mel_results_path=None):
    """Print a side-by-side comparison table. Loads mel results from JSON if available."""
    mel_results = None
    if mel_results_path and os.path.exists(mel_results_path):
        with open(mel_results_path) as f:
            data = json.load(f)
        mel_results = data.get("fold_results", [])

    metrics = ["accuracy", "f1_weighted", "f1_macro", "auc_roc", "balanced_accuracy", "mcc"]
    labels  = ["Accuracy", "F1 (weighted)", "F1 (macro)", "AUC-ROC", "Balanced Acc", "MCC"]

    sep = "=" * 74
    logging.info(f"\n{sep}")
    logging.info("COMPARISON: Wav2Vec2 Baseline  vs.  Mel-Spectrogram CNN-Transformer")
    logging.info(sep)

    if mel_results:
        logging.info(f"{'Metric':<18} | {'Wav2Vec2':>10} ± {'std':>6} | {'Mel CNN-T':>10} ± {'std':>6} | {'Δ (Mel−W2V)':>11}")
        logging.info("-" * 74)
        for metric, label in zip(metrics, labels):
            w2v_vals = [r[metric] for r in w2v_results]
            mel_vals = [r[metric] for r in mel_results]
            w2v_m, w2v_s = np.mean(w2v_vals), np.std(w2v_vals)
            mel_m, mel_s = np.mean(mel_vals), np.std(mel_vals)
            delta = mel_m - w2v_m
            arrow = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "~")
            logging.info(
                f"{label:<18} | {w2v_m:>10.4f} ± {w2v_s:>6.4f} | "
                f"{mel_m:>10.4f} ± {mel_s:>6.4f} | {delta:>+.4f} {arrow}"
            )
    else:
        logging.info(f"{'Metric':<18} | {'Wav2Vec2':>10} ± {'std':>6}")
        logging.info("-" * 40)
        for metric, label in zip(metrics, labels):
            vals = [r[metric] for r in w2v_results]
            logging.info(f"{label:<18} | {np.mean(vals):>10.4f} ± {np.std(vals):>6.4f}")
        logging.info("\n  (Run run_all_folds.py first to generate cv_results.json for Mel comparison)")

    logging.info(sep)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Wav2Vec2 baseline — stress classification")
    parser.add_argument("--label", default="affect3-class",
                        choices=["binary-stress", "affect3-class"])
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS,
                        help="Max audio clip length in seconds (default: 30)")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, just evaluate existing checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    num_classes = (config.NUM_CLASSES_BINARY if args.label == "binary-stress"
                   else config.NUM_CLASSES_AFFECT3)

    logging.info(f"Model:        {W2V_MODEL_ID} (frozen backbone)")
    logging.info(f"Label:        {args.label}  ({num_classes} classes)")
    logging.info(f"Max seconds:  {args.max_seconds}  ({args.max_seconds * SAMPLE_RATE:,} samples)")
    logging.info(f"Device:       {DEVICE}")
    logging.info(f"Folds:        {config.NUM_FOLDS}")

    if not args.eval_only:
        for fold in range(config.NUM_FOLDS):
            train_fold(fold, args.label, num_classes, args.max_seconds,
                       epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    all_results = []
    for fold in range(config.NUM_FOLDS):
        result = eval_fold(fold, args.label, num_classes, args.max_seconds)
        all_results.append(result)

    # Summary
    metrics = ["accuracy", "f1_weighted", "f1_macro", "auc_roc", "balanced_accuracy", "mcc"]
    names   = ["Accuracy", "F1 (weighted)", "F1 (macro)", "AUC-ROC", "Balanced Acc", "MCC"]
    logging.info(f"\n{'='*70}\n5-FOLD CV SUMMARY — Wav2Vec2 Baseline ({args.label})\n{'='*70}")
    for r in all_results:
        logging.info(
            f"  Fold {r['fold']}: acc={r['accuracy']:.4f} | F1w={r['f1_weighted']:.4f} | "
            f"F1m={r['f1_macro']:.4f} | AUC={r['auc_roc']:.4f} | "
            f"BalAcc={r['balanced_accuracy']:.4f} | MCC={r['mcc']:.4f}"
        )
    logging.info("-" * 70)
    for metric, name in zip(metrics, names):
        vals = [r[metric] for r in all_results]
        logging.info(f"  {name:<16}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    logging.info("=" * 70)

    # Comparison table vs. mel results (if available)
    mel_json = os.path.join(config.BASE_DIR, "cv_results.json")
    print_comparison(all_results, mel_results_path=mel_json)

    # Save wav2vec results
    out_path = os.path.join(config.BASE_DIR, "wav2vec_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": W2V_MODEL_ID,
            "label": args.label,
            "max_seconds": args.max_seconds,
            "n_folds": config.NUM_FOLDS,
            "held_out_subjects": config.HELD_OUT_SUBJECTS,
            "fold_results": all_results,
        }, f, indent=2, default=str)
    logging.info(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    log_path = os.path.join(config.BASE_DIR, "wav2vec_results.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Wav2Vec2 Baseline — logs → {log_path}\n")
    main()
