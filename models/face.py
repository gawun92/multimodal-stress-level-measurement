"""
model.py  —  CNN-LSTM Stress Classification (Face Modality)

Usage:
    python model.py                                    # default: kfold=N, epochs=130
    python model.py --kfold Y                          # K-Fold (10 splits)
    python model.py --kfold N                          # train on all data, eval on held-out
    python model.py --kfold Y --epochs 50
    python model.py --label affect3-class --kfold Y    # 3-class classification
"""

import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Callable
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# ─────────────────Config─────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data/stressid/labels.csv")
FACE_DIR = os.path.join(BASE_DIR, "feature_extraction/results/face/train")
TEST_IDS: List[str] = ["wssm", "x1q3", "y8c3", "y9z6"]
# ────────────────────────────────────────

MAX_FRAMES = 300
N_LANDMARKS = 86
N_COORDS = 3

# Hyperparameters
INPUT_SIZE = N_LANDMARKS * N_COORDS
CNN_CHANNELS = [256, 128]
KERNEL_SIZE = 3
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
N_SPLITS = 10
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

class StressDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, int]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, label = self.samples[idx]
        x = arr.reshape(MAX_FRAMES, N_LANDMARKS * N_COORDS)
        x = torch.from_numpy(x)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

# parse CSV
def load_stress_labels(csv_path: str = CSV_PATH, label_col: str = "binary-stress") -> Dict[str, int]:
    """Load labels.csv -> {subject_id: class_label}."""
    label_map: Dict[str, int] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            subject_id = row["subject/task"].split("_")[0]
            label_map[subject_id] = int(row[label_col])
    return label_map


# sample builder
def build_subject_samples(
        face_dir: str = FACE_DIR,
        csv_path: str = CSV_PATH,
        label_col: str = "binary-stress",
        test_ids: List[str] = TEST_IDS,
) -> Tuple[Dict[str, List], list]:
    label_map = load_stress_labels(csv_path, label_col)
    kfold_subjects: Dict[str, List[Tuple[np.ndarray, int]]] = {}
    held_out_samples = []

    for pid in os.listdir(face_dir):
        pid_dir = os.path.join(face_dir, pid)
        if not os.path.isdir(pid_dir):
            continue
        if pid not in label_map:
            continue

        label = label_map[pid]
        is_held_out = pid in test_ids

        for npy_file in sorted(os.listdir(pid_dir)):
            if not npy_file.endswith("_face.npy"):
                continue
            npy_path = os.path.join(pid_dir, npy_file)
            arr = np.load(npy_path).astype(np.float32)

            if is_held_out:
                held_out_samples.append((arr, label, pid, npy_file))
            else:
                kfold_subjects.setdefault(pid, []).append((arr, label))

    kfold_total = sum(len(v) for v in kfold_subjects.values())
    print(f"[dataset] K-Fold subjects  : {len(kfold_subjects)} ({kfold_total} samples)")
    print(f"[dataset] Held-out subjects: {len(test_ids)} ({len(held_out_samples)} samples)")
    return kfold_subjects, held_out_samples

def get_kfold_dataloaders(
        fold_idx: int,
        kfold_subjects: Dict[str, List],
        n_splits: int = N_SPLITS,
        batch_size: int = BATCH_SIZE,
        seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    rng = np.random.default_rng(seed)
    all_pids = list(rng.permutation(sorted(kfold_subjects.keys())))
    chunks = np.array_split(all_pids, n_splits)

    val_pids = list(chunks[fold_idx])
    trn_pids = [p for i, c in enumerate(chunks) for p in c if i != fold_idx]

    def flatten(pids):
        out = []
        for pid in pids:
            out.extend(kfold_subjects[pid])
        return out

    train_loader = DataLoader(StressDataset(flatten(trn_pids)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(StressDataset(flatten(val_pids)),
                            batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, val_pids

def compute_metrics(preds: np.ndarray, labels: np.ndarray,
                    probs: np.ndarray = None) -> Dict[str, float]:
    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_m = f1_score(labels, preds, average="macro", zero_division=0)
    auc = float("nan")
    if probs is not None and len(np.unique(labels)) > 1:
        if probs.shape[1] == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class="ovr")
    return {"Accuracy": acc, "Weighted_F1": f1_w, "Macro_F1": f1_m, "AUC_ROC": auc}

# evaluation
def evaluate_on_heldout(
        model,
        held_out_samples: list,
        label_col: str = "binary-stress",
        model_name: str = "CNN-LSTM",
        device: str = DEVICE,
) -> Dict[str, float]:
    model.eval()
    model.to(device)

    print(f"\n{'=' * 65}")
    print(f"  [{model_name}]  HELD-OUT FINAL TEST  (subjects: {TEST_IDS})")
    print(f"{'=' * 65}")
    print(f"  {'Subject':<10} {'File':<28} {'Actual':>7} {'Pred':>7}")
    print(f"  {'-' * 56}")

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for arr, label, pid, npy_file in held_out_samples:
            x = torch.from_numpy(
                arr.reshape(1, MAX_FRAMES, N_LANDMARKS * N_COORDS)
            ).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            pred = logits.argmax(1).item()
            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(probs[0])
            print(f"  {pid:<10} {npy_file:<28} {label:>7} {pred:>7}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    print(f"  {'─' * 56}")
    metrics = compute_metrics(all_preds, all_labels, all_probs)
    print(f"  Accuracy:    {metrics['Accuracy']:.4f}")
    print(f"  Weighted F1: {metrics['Weighted_F1']:.4f}")
    print(f"  Macro F1:    {metrics['Macro_F1']:.4f}")
    print(f"  AUC-ROC:     {metrics['AUC_ROC']:.4f}")

    if label_col == "binary-stress":
        target_names = ["no-stress", "stressed"]
    else:
        target_names = ["class-0", "class-1", "class-2"]
    report = classification_report(all_labels, all_preds,
                                   target_names=target_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n{report}")
    print(f"  Confusion Matrix:\n{cm}")

    print(f"{'=' * 65}")
    return metrics

def run_kfold(
        model_fn: Callable,
        train_fn: Callable,
        label_col: str = "binary-stress",
        n_splits: int = N_SPLITS,
        batch_size: int = BATCH_SIZE,
        seed: int = 42,
        **train_kwargs,
) -> Tuple[List[Dict], list, list]:
    kfold_subjects, held_out_samples = build_subject_samples(label_col=label_col)
    fold_results = []
    fold_models = []

    for fold in range(n_splits):
        print(f"\n{'=' * 60}")
        print(f"  [CNN-LSTM]  Fold {fold + 1} / {n_splits}")
        print(f"{'=' * 60}")

        train_loader, val_loader, val_pids = get_kfold_dataloaders(
            fold_idx=fold, kfold_subjects=kfold_subjects,
            n_splits=n_splits, batch_size=batch_size, seed=seed,
        )
        print(f"  val subjects : {val_pids}")
        print(f"  train        : {len(train_loader.dataset)} samples  |  "
              f"val: {len(val_loader.dataset)} samples")

        model = model_fn()
        metrics, trained_model = train_fn(
            model=model, train_loader=train_loader,
            val_loader=val_loader, fold=fold, **train_kwargs,
        )
        fold_results.append(metrics)
        fold_models.append(trained_model)

        print(f"\n  ┌─ Fold {fold + 1} Best Val Results")
        print(f"  │  Accuracy:    {metrics['Accuracy']:.4f}")
        print(f"  │  Weighted F1: {metrics['Weighted_F1']:.4f}")
        print(f"  │  Macro F1:    {metrics['Macro_F1']:.4f}")
        print(f"  │  AUC-ROC:     {metrics['AUC_ROC']:.4f}")
        print(f"  └{'─' * 43}")

    return fold_results, held_out_samples, fold_models

# model
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        w = F.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


class StressCNNLSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, cnn_channels=CNN_CHANNELS,
                 kernel_size=KERNEL_SIZE, lstm_hidden=LSTM_HIDDEN,
                 lstm_layers=LSTM_LAYERS, dropout=DROPOUT, num_classes=2):
        super().__init__()
        cnn_layers = []
        in_ch = input_size
        for out_ch in cnn_channels:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        self.lstm = nn.LSTM(input_size=cnn_channels[-1], hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.attn_pool = AttentionPooling(lstm_hidden * 2)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x):
        B, T, F = x.shape
        x = x.view(B * T, F, 1)
        x = self.cnn(x).mean(dim=-1)
        x = x.view(B, T, -1)
        x, _ = self.lstm(x)
        x = self.attn_pool(x)
        return self.head(x)

def train_one_fold(model, train_loader, val_loader,
                   fold=0, epochs=EPOCHS, lr=LR, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"cnn_lstm_fold{fold}.pt")

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_metrics = {"Accuracy": 0.0, "Weighted_F1": 0.0, "Macro_F1": 0.0, "AUC_ROC": 0.0}
    best_state = None

    print(f"\n  {'Epoch':>6}  {'TrainLoss':>10}  {'Acc':>8}  {'wF1':>8}  {'mF1':>8}")
    print(f"  {'-' * 52}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_loader.dataset)

        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(DEVICE))
                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
                all_probs.extend(probs)

        metrics = compute_metrics(np.array(all_preds), np.array(all_labels),
                                  np.array(all_probs))
        scheduler.step(train_loss)

        print(f"  {epoch:>6}  {train_loss:>10.4f}  "
              f"{metrics['Accuracy']:>8.4f}  {metrics['Weighted_F1']:>8.4f}  "
              f"{metrics['Macro_F1']:>8.4f}")

        if metrics["Accuracy"] > best_metrics["Accuracy"]:
            best_metrics = metrics
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)

    model.load_state_dict(best_state)
    return best_metrics, model

def train_full(label_col="binary-stress", epochs=EPOCHS, lr=LR,
               batch_size=BATCH_SIZE, save_dir="checkpoints"):
    num_classes = 2 if label_col == "binary-stress" else 3
    kfold_subjects, held_out_samples = build_subject_samples(label_col=label_col)

    all_samples = []
    for samples in kfold_subjects.values():
        all_samples.extend(samples)

    train_loader = DataLoader(StressDataset(all_samples),
                              batch_size=batch_size, shuffle=True)

    print(f"\n{'=' * 60}")
    print(f"  [CNN-LSTM]  Full Training (no K-Fold)  task={label_col}")
    print(f"{'=' * 60}")
    print(f"  train    : {len(all_samples)} samples  (held-out excluded)")
    print(f"  held-out : {len(held_out_samples)} samples → final eval only")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"cnn_lstm_full_{label_col}.pt")

    model = StressCNNLSTM(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"\n  {'Epoch':>6}  {'TrainLoss':>10}")
    print(f"  {'-' * 22}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(all_samples)

        print(f"  {epoch:>6}  {train_loss:>10.4f}")
        scheduler.step(train_loss)

    torch.save(model.state_dict(), save_path)
    print(f"\n  Model saved → {save_path}")

    evaluate_on_heldout(model, held_out_samples, label_col=label_col)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kfold", type=str, default="N", choices=["Y", "N"],
                        help="Y: K-Fold training  |  N: train on all data")
    parser.add_argument("--label", type=str, default="binary-stress",
                        choices=["binary-stress", "affect3-class"],
                        help="Label column from labels.csv")
    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--save_dir", default="checkpoints")
    args = parser.parse_args()

    num_classes = 2 if args.label == "binary-stress" else 3

    if args.kfold == "Y":
        fold_results, held_out_samples, fold_models = run_kfold(
            model_fn=lambda: StressCNNLSTM(num_classes=num_classes),
            train_fn=train_one_fold,
            label_col=args.label,
            n_splits=args.n_splits,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=args.save_dir,
        )
        # K-Fold summary
        keys = ["Accuracy", "Weighted_F1", "Macro_F1", "AUC_ROC"]
        print("\n" + "=" * 60)
        print(f"  K-FOLD CROSS-VALIDATION SUMMARY  (task={args.label})")
        print("=" * 60)
        for k in keys:
            vals = np.array([r[k] for r in fold_results])
            print(f"  {k:<14}  mean={np.nanmean(vals):.4f}  std={np.nanstd(vals):.4f}")
        print("=" * 60)

        best_idx = max(range(len(fold_results)),
                       key=lambda i: fold_results[i]["Accuracy"])
        best_model = fold_models[best_idx]
        print(f"\n  Best fold: {best_idx + 1}  "
              f"(Accuracy={fold_results[best_idx]['Accuracy']:.4f})")
        evaluate_on_heldout(best_model, held_out_samples, label_col=args.label)

    else:
        train_full(label_col=args.label, epochs=args.epochs, lr=args.lr,
                   batch_size=args.batch_size, save_dir=args.save_dir)