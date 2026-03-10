"""
model.py  —  CNN-LSTM Stress Prediction (Face Modality)

Usage:
    python model.py                 # default: kfold=N, epochs=130
    python model.py --kfold Y       # K-Fold (10 splits)
    python model.py --kfold N       # train on all data, eval on held-out
    python model.py --kfold Y --epochs 50
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

# ─────────────────Config─────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data/stressid/train/self_assessments.csv")
FACE_DIR = os.path.join(BASE_DIR, "feature_extraction/results/face/train")
TEST_IDS: List[str] = ["wssm", "x1q3", "y8c3", "y9z6"]
# ────────────────────────────────────────

# 11 stress tasks
STRESS_TASKS: List[str] = [
    "Breathing_stress", "Video1_stress", "Video2_stress",
    "Counting1_stress", "Counting2_stress", "Stroop_stress",
    "Speaking_stress", "Math_stress", "Reading_stress",
    "Counting3_stress", "Relax_stress",
]

# mapping task name to npy file
TASK_TO_STEM: Dict[str, str] = {
    "Breathing_stress": "Breathing",
    "Video1_stress": "Video1",
    "Video2_stress": "Video2",
    "Counting1_stress": "Counting1",
    "Counting2_stress": "Counting2",
    "Stroop_stress": "Stroop",
    "Speaking_stress": "Speaking",
    "Math_stress": "Math",
    "Reading_stress": "Reading",
    "Counting3_stress": "Counting3",
    "Relax_stress": "Relax",
}

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
    def __init__(self, samples: List[Tuple[np.ndarray, float]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, score = self.samples[idx]
        x = arr.reshape(MAX_FRAMES, N_LANDMARKS * N_COORDS)
        x = torch.from_numpy(x)
        y = torch.tensor(score, dtype=torch.float32)
        return x, y

# parse CSV
def load_stress_labels(csv_path: str = CSV_PATH) -> Dict[str, Dict[str, float]]:
    labels: Dict[str, Dict[str, float]] = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter=";"))

    participant_ids = rows[0][1:]
    for pid in participant_ids:
        labels[pid] = {}

    for row in rows[1:]:
        task_name = row[0].strip()
        if task_name not in STRESS_TASKS:
            continue
        for col_idx, pid in enumerate(participant_ids):
            raw = row[col_idx + 1].strip()
            if not raw:
                continue
            try:
                labels[pid][task_name] = float(raw)
            except ValueError:
                pass
    return labels


# sample builder
def build_subject_samples(
        face_dir: str = FACE_DIR,
        csv_path: str = CSV_PATH,
        test_ids: List[str] = TEST_IDS,
) -> Tuple[Dict[str, List], list]:
    labels = load_stress_labels(csv_path)
    kfold_subjects: Dict[str, List[Tuple[np.ndarray, float]]] = {}
    held_out_samples = []
    missing = 0

    for pid, task_scores in labels.items():
        is_held_out = pid in test_ids
        bucket = []

        for task_name, score in task_scores.items():
            stem = TASK_TO_STEM.get(task_name)
            if stem is None:
                continue
            npy_path = os.path.join(face_dir, pid, f"{stem}_face.npy")
            if not os.path.exists(npy_path):
                missing += 1
                continue
            arr = np.load(npy_path).astype(np.float32)

            if is_held_out:
                held_out_samples.append((arr, score, pid, task_name))
            else:
                bucket.append((arr, score))

        if not is_held_out and bucket:
            kfold_subjects[pid] = bucket

    kfold_total = sum(len(v) for v in kfold_subjects.values())
    print(f"[dataset] K-Fold subjects  : {len(kfold_subjects)} ({kfold_total} samples)")
    print(f"[dataset] Held-out subjects: {len(test_ids)} ({len(held_out_samples)} samples)")
    print(f"[dataset] Missing npy      : {missing}")
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

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((preds - targets) ** 2))
    rmse = float(np.sqrt(mse))
    pearson = float(np.corrcoef(preds, targets)[0, 1]) \
        if preds.std() > 1e-8 and targets.std() > 1e-8 else 0.0
    return {"MSE": mse, "RMSE": rmse, "Pearson": pearson}

# evaluation
def evaluate_on_heldout(
        model,
        held_out_samples: list,
        model_name: str = "CNN-LSTM",
        device: str = DEVICE,
) -> Dict[str, float]:
    model.eval()
    model.to(device)

    print(f"\n{'=' * 65}")
    print(f"  [{model_name}]  HELD-OUT FINAL TEST  (subjects: {TEST_IDS})")
    print(f"{'=' * 65}")
    print(f"  {'Subject':<10} {'Task':<22} {'Actual':>7} {'Pred':>7} {'Error':>8}")
    print(f"  {'-' * 56}")

    all_preds, all_targets = [], []

    with torch.no_grad():
        for arr, score, pid, task_name in held_out_samples:
            x = torch.from_numpy(
                arr.reshape(1, MAX_FRAMES, N_LANDMARKS * N_COORDS)
            ).to(device)
            pred = model(x).item()
            error = pred - score
            all_preds.append(pred)
            all_targets.append(score)
            task_short = task_name.replace("_stress", "")
            print(f"  {pid:<10} {task_short:<22} {score:>7.1f} {pred:>7.2f} {error:>+8.2f}")

    print(f"  {'─' * 56}")
    metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
    print(f"  Overall  →  RMSE={metrics['RMSE']:.4f}  Pearson={metrics['Pearson']:.4f}")

    # Tolerance accuracy
    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    print(f"\n  Tolerance Accuracy:")
    for tol in [0.5, 1.0, 1.5, 2.0]:
        acc = float(np.mean(np.abs(preds_arr - targets_arr) <= tol)) * 100
        print(f"    ±{tol:.1f}  →  {acc:.1f}%")

    print(f"{'=' * 65}")
    return metrics

def run_kfold(
        model_fn: Callable,
        train_fn: Callable,
        n_splits: int = N_SPLITS,
        batch_size: int = BATCH_SIZE,
        seed: int = 42,
        **train_kwargs,
) -> Tuple[List[Dict], list, list]:
    kfold_subjects, held_out_samples = build_subject_samples()
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
        print(f"  │  RMSE    : {metrics['RMSE']:.4f}")
        print(f"  │  Pearson : {metrics['Pearson']:.4f}")
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
                 lstm_layers=LSTM_LAYERS, dropout=DROPOUT):
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
            nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        B, T, F = x.shape
        x = x.view(B * T, F, 1)
        x = self.cnn(x).mean(dim=-1)
        x = x.view(B, T, -1)
        x, _ = self.lstm(x)
        x = self.attn_pool(x)
        return self.head(x).squeeze(-1)

def train_one_fold(model, train_loader, val_loader,
                   fold=0, epochs=EPOCHS, lr=LR, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"cnn_lstm_fold{fold}.pt")

    model = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_metrics = {"RMSE": float("inf"), "Pearson": 0.0, "MSE": float("inf")}
    best_state = None

    print(f"\n  {'Epoch':>6}  {'TrainLoss':>10}  {'RMSE':>8}  {'Pearson':>8}")
    print(f"  {'-' * 46}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_loader.dataset)

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                preds.extend(model(x.to(DEVICE)).cpu().numpy().tolist())
                targets.extend(y.numpy().tolist())

        metrics = compute_metrics(np.array(preds), np.array(targets))
        scheduler.step(metrics["RMSE"])

        print(f"  {epoch:>6}  {train_loss:>10.4f}  "
              f"{metrics['RMSE']:>8.4f}  {metrics['Pearson']:>8.4f}")

        if metrics["RMSE"] < best_metrics["RMSE"]:
            best_metrics = metrics
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)

    model.load_state_dict(best_state)
    return best_metrics, model

def train_full(epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, save_dir="checkpoints"):
    kfold_subjects, held_out_samples = build_subject_samples()

    all_samples = []
    for samples in kfold_subjects.values():
        all_samples.extend(samples)

    train_loader = DataLoader(StressDataset(all_samples),
                              batch_size=batch_size, shuffle=True)

    print(f"\n{'=' * 60}")
    print(f"  [CNN-LSTM]  Full Training (no K-Fold)")
    print(f"{'=' * 60}")
    print(f"  train    : {len(all_samples)} samples  (held-out excluded)")
    print(f"  held-out : {len(held_out_samples)} samples → final eval only")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cnn_lstm_full.pt")

    model = StressCNNLSTM().to(DEVICE)
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
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(all_samples)

        print(f"  {epoch:>6}  {train_loss:>10.4f}")
        scheduler.step(train_loss)

    torch.save(model.state_dict(), save_path)
    print(f"\n  Model saved → {save_path}")

    evaluate_on_heldout(model, held_out_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kfold", type=str, default="N", choices=["Y", "N"],
                        help="Y: K-Fold training  |  N: train on all data")
    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--save_dir", default="checkpoints")
    args = parser.parse_args()

    if args.kfold == "Y":
        fold_results, held_out_samples, fold_models = run_kfold(
            model_fn=StressCNNLSTM,
            train_fn=train_one_fold,
            n_splits=args.n_splits,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=args.save_dir,
        )
        # K-Fold summary
        keys = ["RMSE", "Pearson"]
        print("\n" + "=" * 60)
        print("  K-FOLD CROSS-VALIDATION SUMMARY")
        print("=" * 60)
        for k in keys:
            vals = np.array([r[k] for r in fold_results])
            print(f"  {k:<10}  mean={vals.mean():.4f}  std={vals.std():.4f}")
        print("=" * 60)

        best_idx = min(range(len(fold_results)), key=lambda i: fold_results[i]["RMSE"])
        best_model = fold_models[best_idx]
        print(f"\n  ★ Best fold: {best_idx + 1}  (RMSE={fold_results[best_idx]['RMSE']:.4f})")
        evaluate_on_heldout(best_model, held_out_samples)

    else:
        train_full(epochs=args.epochs, lr=args.lr,
                   batch_size=args.batch_size, save_dir=args.save_dir)
