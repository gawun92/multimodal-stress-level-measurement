import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Callable


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, "data/stressid/train/self_assessments.csv")
FACE_DIR = os.path.join(BASE_DIR, "feature_extraction/results/face/train")

# Held-out test subjects
TEST_IDS: List[str] = ["wssm", "x1q3", "y8c3", "y9z6"]

# 11 stress tasks
STRESS_TASKS: List[str] = [
    "Breathing_stress", "Video1_stress",  "Video2_stress",
    "Counting1_stress", "Counting2_stress","Stroop_stress",
    "Speaking_stress",  "Math_stress",    "Reading_stress",
    "Counting3_stress", "Relax_stress",
]

# Maps task name to npy file stem
TASK_TO_STEM: Dict[str, str] = {
    "Breathing_stress" : "Breathing",
    "Video1_stress"    : "Video1",
    "Video2_stress"    : "Video2",
    "Counting1_stress" : "Counting1",
    "Counting2_stress" : "Counting2",
    "Stroop_stress"    : "Stroop",
    "Speaking_stress"  : "Speaking",
    "Math_stress"      : "Math",
    "Reading_stress"   : "Reading",
    "Counting3_stress" : "Counting3",
    "Relax_stress"     : "Relax",
}

MAX_FRAMES  = 300
N_LANDMARKS = 86   # key landmarks only (eyes, eyebrows, mouth full contour)
N_COORDS    = 3

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


def build_subject_samples(
        face_dir : str = FACE_DIR,
        csv_path : str = CSV_PATH,
        test_ids : List[str] = TEST_IDS,
) -> Tuple[Dict[str, List], list]:
    labels  = load_stress_labels(csv_path)
    kfold_subjects: Dict[str, List[Tuple[np.ndarray, float]]] = {}
    held_out_samples = []
    missing = 0

    for pid, task_scores in labels.items():
        is_held_out = pid in test_ids
        bucket = []

        for task_name, score in task_scores.items():
            stem     = TASK_TO_STEM.get(task_name)
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
        fold_idx       : int,
        kfold_subjects : Dict[str, List],
        n_splits       : int = 5,
        batch_size     : int = 16,
        seed           : int = 42,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    rng      = np.random.default_rng(seed)
    all_pids = list(rng.permutation(sorted(kfold_subjects.keys())))
    chunks   = np.array_split(all_pids, n_splits)

    val_pids = list(chunks[fold_idx])
    trn_pids = [p for i, c in enumerate(chunks) for p in c if i != fold_idx]

    def flatten(pids):
        out = []
        for pid in pids:
            out.extend(kfold_subjects[pid])
        return out

    train_loader = DataLoader(StressDataset(flatten(trn_pids)),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(StressDataset(flatten(val_pids)),
                              batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, val_pids

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    mse     = float(np.mean((preds - targets) ** 2))
    rmse    = float(np.sqrt(mse))

    pearson = float(np.corrcoef(preds, targets)[0, 1]) \
        if preds.std() > 1e-8 and targets.std() > 1e-8 else 0.0
    return {"MSE": mse, "RMSE": rmse, "Pearson": pearson}

def run_kfold(
        model_fn       : Callable,
        train_fn       : Callable,
        face_dir       : str = FACE_DIR,
        csv_path       : str = CSV_PATH,
        test_ids       : List[str] = TEST_IDS,
        n_splits       : int = 5,
        n_folds_to_run : int = None,
        batch_size     : int = 16,
        seed           : int = 42,
        model_name     : str = "Model",
        **train_kwargs,
) -> Tuple[List[Dict], list, list]:
    kfold_subjects, held_out_samples = build_subject_samples(face_dir, csv_path, test_ids)
    fold_results  = []
    fold_models   = []
    fold_histories = []

    total_folds = n_folds_to_run if n_folds_to_run is not None else n_splits

    for fold in range(total_folds):
        print(f"\n{'='*60}")
        print(f"  [{model_name}]  Fold {fold + 1} / {n_splits}  (running {total_folds} fold(s))")
        print(f"{'='*60}")

        train_loader, val_loader, val_pids = get_kfold_dataloaders(
            fold_idx       = fold,
            kfold_subjects = kfold_subjects,
            n_splits       = n_splits,
            batch_size     = batch_size,
            seed           = seed,
        )
        print(f"  val subjects : {val_pids}")
        print(f"  train        : {len(train_loader.dataset)} samples  |  "
              f"val: {len(val_loader.dataset)} samples")

        model = model_fn()
        result = train_fn(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            fold         = fold,
            **train_kwargs,
        )
        # support both (metrics, model) and (metrics, model, history)
        if len(result) == 3:
            metrics, trained_model, history = result
        else:
            metrics, trained_model = result
            history = None

        fold_results.append(metrics)
        fold_models.append(trained_model)
        fold_histories.append(history)

        print(f"\n  ┌─ Fold {fold+1} Best Val Results {'─'*28}")
        print(f"  │  RMSE    : {metrics['RMSE']:.4f}")
        print(f"  │  Pearson : {metrics['Pearson']:.4f}")
        print(f"  └{'─'*43}")

    return fold_results, held_out_samples, fold_models, fold_histories

def evaluate_on_heldout(
        model,
        held_out_samples : list,
        model_name       : str = "Model",
        device           : str = "cpu",
) -> Dict[str, float]:
    model.eval()
    model.to(device)

    print(f"\n{'='*65}")
    print(f"  [{model_name}]  HELD-OUT FINAL TEST  (subjects: {TEST_IDS})")
    print(f"{'='*65}")
    print(f"  {'Subject':<10} {'Task':<22} {'Actual':>7} {'Pred':>7} {'Error':>8}")
    print(f"  {'-'*56}")

    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for arr, score, pid, task_name in held_out_samples:
            x    = torch.from_numpy(
                arr.reshape(1, MAX_FRAMES, N_LANDMARKS * N_COORDS)
            ).to(device)
            pred  = model(x).item()
            error = pred - score
            all_preds.append(pred)
            all_targets.append(score)

            task_short = task_name.replace("_stress", "")
            print(f"  {pid:<10} {task_short:<22} {score:>7.1f} {pred:>7.2f} {error:>+8.2f}")

    print(f"  {'─'*56}")
    metrics = compute_metrics(np.array(all_preds), np.array(all_targets))
    print(f"  Overall  →  RMSE={metrics['RMSE']:.4f}  "
          f"Pearson={metrics['Pearson']:.4f}")
    print(f"{'='*65}")
    return metrics

def print_kfold_summary(results_by_model: Dict[str, List[Dict]]) -> None:
    keys = ["RMSE", "Pearson"]
    print("\n" + "=" * 70)
    print("  K-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<14}", end="")
    for k in keys:
        print(f"  {k+' mean':>11}  {k+' std':>9}", end="")
    print()
    print("-" * 70)
    for model_name, fold_results in results_by_model.items():
        print(f"  {model_name:<14}", end="")
        for k in keys:
            vals = np.array([r[k] for r in fold_results])
            print(f"  {vals.mean():>11.4f}  {vals.std():>9.4f}", end="")
        print()
    print("=" * 70)
    best = min(results_by_model.items(),
               key=lambda kv: np.mean([r["RMSE"] for r in kv[1]]))
    print(f"\n  ★ Best model (K-Fold RMSE): {best[0]}")
    print(f"    Mean RMSE = {np.mean([r['RMSE'] for r in best[1]]):.4f}\n")

def print_heldout_summary(heldout_by_model: Dict[str, Dict]) -> None:
    keys = ["RMSE", "Pearson"]
    print("\n" + "=" * 70)
    print("  FINAL HELD-OUT TEST SUMMARY")
    print(f"  Held-out subjects : {TEST_IDS}")
    print("=" * 70)
    print(f"  {'Model':<14}", end="")
    for k in keys:
        print(f"  {k:>11}", end="")
    print()
    print("-" * 70)
    for model_name, metrics in heldout_by_model.items():
        print(f"  {model_name:<14}", end="")
        for k in keys:
            print(f"  {metrics[k]:>11.4f}", end="")
        print()
    print("=" * 70)
    best = min(heldout_by_model.items(), key=lambda kv: kv[1]["RMSE"])
    print(f"\n  ★ Best model (Held-out RMSE): {best[0]}")
    print(f"    RMSE = {best[1]['RMSE']:.4f}\n")