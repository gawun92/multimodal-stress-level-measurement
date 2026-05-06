"""
benchmark_gesture.py

Benchmark a trained gesture classifier on a chosen dataset split.

The canonical gesture modality in this repo is upper-body/head motion.

Example usage:
    python benchmark_gesture.py --fold 0 --label binary-stress
    python benchmark_gesture.py --fold 0 --label affect3-class --split val
    python benchmark_gesture.py --gesture-dir feature_extraction/results/gesture/train --require-mask --n-folds 3 --fold 0 --split test --label binary-stress
    
    python benchmark_gesture.py \
        --gesture-dir feature_extraction/results/upper_body/train_20260416_193709 \
        --label binary-stress \
        --fold 0 \
        --n-folds 3 \
        --split test \
        --require-mask \
        --window-len 96 \
        --min-valid-frames 24 \
        --checkpoint-path checkpoints/gesture_branch_fold0_binary-stress_20260417_153259.pt
"""

import argparse
import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

import config
from dataset import StressGestureDataset, get_all_gesture_subjects, get_subject_splits
from models.gesture_branch import GestureBranch, GestureClassifier


RESULTS_DIR = os.path.join(config.BASE_DIR, "results", "gesture")
FIGURES_DIR = os.path.join(config.BASE_DIR, "figures")


def detect_gesture_schema(gesture_dir):
    for subject_id in get_all_gesture_subjects(gesture_dir):
        subject_dir = os.path.join(gesture_dir, subject_id)
        for filename in sorted(os.listdir(subject_dir)):
            if not filename.endswith("_gesture.npy"):
                continue
            npy_path = os.path.join(subject_dir, filename)
            arr = np.load(npy_path, mmap_mode="r")
            if arr.ndim != 3:
                raise ValueError(
                    f"Expected gesture feature tensor with shape (T, L, C), got {arr.shape} "
                    f"in {npy_path}"
                )
            _, joint_count, coord_dim = arr.shape
            input_dim = joint_count * coord_dim
            return joint_count, coord_dim, input_dim, npy_path
    raise RuntimeError(
        f"No *_gesture.npy files found in {gesture_dir}. "
        "Run gesture feature extraction first."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark gesture branch")
    parser.add_argument("--fold", type=int, default=0, help="CV fold index (0-based)")
    parser.add_argument(
        "--n-folds",
        type=int,
        default=config.NUM_FOLDS,
        help="Number of CV folds to use for subject splitting",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="binary-stress",
        choices=["binary-stress", "affect3-class"],
        help="Label column to benchmark on",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--gesture-dir",
        type=str,
        default=config.GESTURE_DIR,
        help="Directory containing extracted gesture features",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include Baseline task if features/labels are present",
    )
    parser.add_argument(
        "--require-mask",
        action="store_true",
        help="Require *_gesture_mask.npy files and use them during benchmarking",
    )
    parser.add_argument(
        "--window-len",
        type=int,
        default=None,
        help="Optional window length to crop each clip to its densest gesture segment",
    )
    parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=1,
        help="Minimum valid gesture frames required inside the selected window",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=config.CHECKPOINT_DIR)
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional explicit path to a gesture checkpoint .pt file",
    )
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=str, default=FIGURES_DIR)
    return parser.parse_args()


def build_dataset(args):
    label_col = args.label
    gesture_tasks = list(config.VIDEO_TASKS)
    if not args.include_baseline:
        gesture_tasks = [task for task in gesture_tasks if task != "Baseline"]

    train_subjects, val_subjects, test_subjects = get_subject_splits(
        fold=args.fold,
        n_folds=args.n_folds,
        subject_fn=lambda: get_all_gesture_subjects(args.gesture_dir),
        tasks=gesture_tasks,
    )
    split_subjects = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }[args.split]
    dataset = StressGestureDataset(
        split_subjects,
        label_col=label_col,
        gesture_dir=args.gesture_dir,
        tasks=gesture_tasks,
        return_mask=args.require_mask,
        require_mask=args.require_mask,
        window_len=args.window_len,
        min_valid_frames=args.min_valid_frames,
    )
    return dataset, split_subjects, gesture_tasks, label_col


def load_model(
    fold,
    label_col,
    checkpoint_dir,
    joint_count,
    coord_dim,
    input_dim,
    checkpoint_path=None,
):
    num_classes = (
        config.NUM_CLASSES_BINARY
        if label_col == "binary-stress"
        else config.NUM_CLASSES_AFFECT3
    )
    branch = GestureBranch(
        input_size=input_dim,
        embed_dim=config.EMBED_DIM,
        joint_count=joint_count,
        coord_dim=coord_dim,
    )
    model = GestureClassifier(branch, num_classes=num_classes)

    ckpt_path = checkpoint_path
    if ckpt_path is None:
        ckpt_path = os.path.join(
            checkpoint_dir,
            f"gesture_branch_fold{fold}_{label_col}.pt",
        )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(
        torch.load(ckpt_path, map_location=config.DEVICE, weights_only=True)
    )
    model = model.to(config.DEVICE)
    model.eval()
    return model, ckpt_path, num_classes


def apply_gesture_mask(X, mask):
    return X * mask.unsqueeze(-1).to(dtype=X.dtype)


def run_benchmark(model, dataloader):
    import torch.nn.functional as F

    all_preds = []
    all_labels = []
    all_probs = []

    total_samples = 0
    total_batches = 0
    total_valid = 0.0

    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                X, mask, y = batch
                X = apply_gesture_mask(X, mask)
                total_valid += mask.float().mean().item()
            else:
                mask = None
                X, y = batch
            X = X.to(config.DEVICE)
            y = y.to(config.DEVICE)
            if mask is not None:
                mask = mask.to(config.DEVICE)

            logits = model(X, mask=mask)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            total_samples += y.size(0)
            total_batches += 1

    elapsed = time.perf_counter() - start_time

    preds = np.concatenate(all_preds) if all_preds else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    probs = np.concatenate(all_probs) if all_probs else np.array([])

    timing = {
        "elapsed_sec": elapsed,
        "samples": total_samples,
        "batches": total_batches,
        "throughput_samples_per_sec": (total_samples / elapsed) if elapsed > 0 else 0.0,
        "avg_latency_ms_per_sample": (elapsed * 1000.0 / total_samples) if total_samples > 0 else 0.0,
        "avg_latency_ms_per_batch": (elapsed * 1000.0 / total_batches) if total_batches > 0 else 0.0,
        "avg_valid_ratio": (total_valid / total_batches) if total_batches > 0 and total_valid > 0 else None,
    }
    return preds, labels, probs, timing


def compute_metrics(labels, preds, probs, num_classes):
    class_ids = list(range(num_classes))
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds, labels=class_ids),
    }

    if num_classes == 2:
        metrics["target_names"] = ["no-stress", "stressed"]
    else:
        metrics["target_names"] = ["class-0", "class-1", "class-2"]

    unique_labels = np.unique(labels)
    if num_classes == 2:
        metrics["auc_roc"] = (
            roc_auc_score(labels, probs[:, 1]) if unique_labels.size > 1 else float("nan")
        )
    else:
        metrics["auc_roc"] = (
            roc_auc_score(labels, probs, multi_class="ovr")
            if unique_labels.size > 1
            else float("nan")
        )

    metrics["classification_report"] = classification_report(
        labels,
        preds,
        labels=class_ids,
        target_names=metrics["target_names"],
        zero_division=0,
    )
    return metrics


def save_text_report(path, args, ckpt_path, subjects, dataset_size, metrics, timing, gesture_tasks, label_col):
    cm = metrics["confusion_matrix"]
    lines = [
        "Gesture Benchmark Report",
        "=" * 60,
        f"Label: {label_col}",
        f"Fold: {args.fold}",
        f"N-Folds: {args.n_folds}",
        f"Split: {args.split}",
        f"Checkpoint: {ckpt_path}",
        f"Device: {config.DEVICE}",
        f"Gesture dir: {args.gesture_dir}",
        f"Mask mode: {'ON' if args.require_mask else 'OFF'}",
        f"Window len: {args.window_len if args.window_len is not None else 'full'}",
        f"Min valid frames: {args.min_valid_frames}",
        f"Tasks: {gesture_tasks}",
        f"Subjects: {len(subjects)}",
        f"Samples: {dataset_size}",
        "",
        "Metrics",
        "-" * 60,
        f"Accuracy:    {metrics['accuracy']:.4f}",
        f"Weighted F1: {metrics['f1_weighted']:.4f}",
        f"Macro F1:    {metrics['f1_macro']:.4f}",
        f"AUC-ROC:     {metrics['auc_roc']:.4f}",
        "",
        "Timing",
        "-" * 60,
        f"Elapsed Time (s):         {timing['elapsed_sec']:.4f}",
        f"Throughput (samples/s):   {timing['throughput_samples_per_sec']:.4f}",
        f"Avg Latency / Sample (ms): {timing['avg_latency_ms_per_sample']:.4f}",
        f"Avg Latency / Batch (ms):  {timing['avg_latency_ms_per_batch']:.4f}",
        f"Avg Valid Ratio:           {timing['avg_valid_ratio']:.4f}" if timing["avg_valid_ratio"] is not None else "Avg Valid Ratio:           n/a",
        "",
        "Confusion Matrix",
        "-" * 60,
        np.array2string(cm),
        "",
        "Classification Report",
        "-" * 60,
        metrics["classification_report"],
    ]

    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))


def save_plot(path, metrics, timing):
    sns.set_theme(style="whitegrid", font_scale=1.05)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metric_names = ["Accuracy", "Weighted F1", "Macro F1", "AUC-ROC"]
    metric_values = [
        metrics["accuracy"],
        metrics["f1_weighted"],
        metrics["f1_macro"],
        metrics["auc_roc"],
    ]
    bars = axes[0].bar(
        metric_names,
        metric_values,
        color=["#2E7D32", "#1565C0", "#EF6C00", "#6A1B9A"],
        edgecolor="black",
        linewidth=0.6,
    )
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Classification Metrics", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, metric_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    cm = metrics["confusion_matrix"].astype(float)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
    sns.heatmap(
        cm_norm,
        annot=metrics["confusion_matrix"],
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        xticklabels=metrics["target_names"],
        yticklabels=metrics["target_names"],
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix", fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    timing_names = ["Total s", "Samples/s", "ms/sample", "ms/batch"]
    timing_values = [
        timing["elapsed_sec"],
        timing["throughput_samples_per_sec"],
        timing["avg_latency_ms_per_sample"],
        timing["avg_latency_ms_per_batch"],
    ]
    timing_bars = axes[2].bar(
        timing_names,
        timing_values,
        color=["#455A64", "#00897B", "#F9A825", "#D84315"],
        edgecolor="black",
        linewidth=0.6,
    )
    axes[2].set_title("Inference Timing", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=20)
    for bar, value in zip(timing_bars, timing_values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(timing_values) * 0.02 if max(timing_values) > 0 else 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle("Gesture Benchmark Summary", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    joint_count, coord_dim, input_dim, schema_path = detect_gesture_schema(args.gesture_dir)
    if (
        joint_count != config.GESTURE_N_LANDMARKS
        or input_dim != config.GESTURE_INPUT_DIM
    ):
        print(
            "[benchmark_gesture] Detected gesture schema override: "
            f"{joint_count} landmarks x {coord_dim} coords = {input_dim} dims "
            f"(from {schema_path})"
        )
    else:
        print(
            f"[benchmark_gesture] Detected gesture schema: "
            f"{joint_count} landmarks x {coord_dim} coords"
        )

    dataset, subjects, gesture_tasks, label_col = build_dataset(args)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No samples found for split='{args.split}'. "
            "Run gesture feature extraction and training first."
        )

    pin_memory = config.DEVICE.type == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model, ckpt_path, num_classes = load_model(
        args.fold,
        label_col,
        args.checkpoint_dir,
        joint_count,
        coord_dim,
        input_dim,
        checkpoint_path=args.checkpoint_path,
    )
    preds, labels, probs, timing = run_benchmark(model, dataloader)
    metrics = compute_metrics(labels, preds, probs, num_classes)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    report_stem = f"benchmark_{ckpt_tag}_{args.split}_{run_stamp}"
    report_path = os.path.join(args.results_dir, f"{report_stem}.txt")
    figure_path = os.path.join(args.figures_dir, f"{report_stem}.png")

    save_text_report(
        report_path,
        args,
        ckpt_path,
        subjects,
        len(dataset),
        metrics,
        timing,
        gesture_tasks,
        label_col,
    )
    save_plot(figure_path, metrics, timing)

    print(f"[benchmark_gesture] Report saved: {report_path}")
    print(f"[benchmark_gesture] Figure saved: {figure_path}")
