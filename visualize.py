"""
visualize.py

Generate presentation-ready charts for the audio branch.

Usage:
    python visualize.py                     # All charts
    python visualize.py --only dataset      # Dataset stats only
    python visualize.py --only spectrogram  # Mel spectrogram examples
    python visualize.py --only training     # Training curves (requires trained model)
    python visualize.py --only attention    # Attention weight visualization
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config
from dataset import StressAudioDataset, get_subject_splits, get_all_audio_subjects
from models.audio_branch import AudioBranch, AudioClassifier

OUT_DIR = os.path.join(config.BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.2)


# ─────────────────────────────────────────
# 1. Dataset Statistics
# ─────────────────────────────────────────
def plot_dataset_stats():
    labels_df = pd.read_csv(config.LABELS_CSV)

    # Filter to audio tasks only
    audio_mask = labels_df["subject/task"].apply(
        lambda x: "_".join(x.split("_")[1:]) in config.AUDIO_TASKS
    )
    audio_df = labels_df[audio_mask].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Binary stress distribution
    counts = audio_df["binary-stress"].value_counts().sort_index()
    colors = ["#4CAF50", "#F44336"]
    bars = axes[0].bar(["No Stress (0)", "Stressed (1)"], counts.values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                     str(val), ha="center", va="bottom", fontweight="bold", fontsize=14)
    axes[0].set_title("Binary Stress Distribution\n(Audio Samples)", fontweight="bold")
    axes[0].set_ylabel("Count")

    # 3-class affect distribution
    counts3 = audio_df["affect3-class"].value_counts().sort_index()
    colors3 = ["#4CAF50", "#FFC107", "#F44336"]
    bars3 = axes[1].bar(["Class 0\n(Low)", "Class 1\n(Medium)", "Class 2\n(High)"],
                        counts3.values, color=colors3, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars3, counts3.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                     str(val), ha="center", va="bottom", fontweight="bold", fontsize=14)
    axes[1].set_title("3-Class Affect Distribution\n(Audio Samples)", fontweight="bold")
    axes[1].set_ylabel("Count")

    # Per-task stress rate
    audio_df["task"] = audio_df["subject/task"].apply(lambda x: "_".join(x.split("_")[1:]))
    task_stress = audio_df.groupby("task")["binary-stress"].mean().sort_values(ascending=False)
    colors_task = ["#F44336" if v > 0.5 else "#4CAF50" for v in task_stress.values]
    bars_t = axes[2].barh(task_stress.index, task_stress.values, color=colors_task, edgecolor="black", linewidth=0.5)
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("Stress Rate")
    axes[2].set_title("Stress Rate by Task", fontweight="bold")
    axes[2].axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    for bar, val in zip(bars_t, task_stress.values):
        axes[2].text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{val:.0%}", va="center", fontsize=11)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "dataset_statistics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 2. Mel Spectrogram Examples
# ─────────────────────────────────────────
def plot_spectrograms():
    labels_df = pd.read_csv(config.LABELS_CSV).set_index("subject/task")
    subjects = get_all_audio_subjects()

    # Find stressed and non-stressed examples
    stressed_examples = []
    nonstress_examples = []

    for subj in subjects:
        for task in config.AUDIO_TASKS:
            key = f"{subj}_{task}"
            npy_path = os.path.join(config.MEL_DIR, subj, f"{task}_mel.npy")
            if key not in labels_df.index or not os.path.exists(npy_path):
                continue
            label = int(labels_df.loc[key, "binary-stress"])
            if label == 1 and len(stressed_examples) < 3:
                stressed_examples.append((key, npy_path))
            elif label == 0 and len(nonstress_examples) < 3:
                nonstress_examples.append((key, npy_path))
            if len(stressed_examples) >= 3 and len(nonstress_examples) >= 3:
                break

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    for i, (key, path) in enumerate(nonstress_examples[:3]):
        mel = np.load(path).squeeze(0)
        axes[0, i].imshow(mel, aspect="auto", origin="lower", cmap="magma")
        axes[0, i].set_title(f"{key}\n[No Stress]", fontsize=11)
        axes[0, i].set_ylabel("Mel Bin") if i == 0 else None
        axes[0, i].set_xlabel("Time Frame")

    for i, (key, path) in enumerate(stressed_examples[:3]):
        mel = np.load(path).squeeze(0)
        axes[1, i].imshow(mel, aspect="auto", origin="lower", cmap="magma")
        axes[1, i].set_title(f"{key}\n[Stressed]", fontsize=11, color="#D32F2F")
        axes[1, i].set_ylabel("Mel Bin") if i == 0 else None
        axes[1, i].set_xlabel("Time Frame")

    fig.suptitle("Mel Spectrogram Examples: No Stress vs. Stressed", fontweight="bold", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "mel_spectrogram_examples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 3. Training Curves (parsed from train.py output log)
# ─────────────────────────────────────────
def plot_training_curves(log_file=None):
    """Parse training output and plot curves. If no log file, run from checkpoint metadata."""
    # Try to find a training log or run a quick training to capture output
    if log_file and os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
    else:
        print("  [training curves] No log file provided. Running training to capture output...")
        import subprocess
        result = subprocess.run(
            ["python", "train.py", "--label", "binary-stress", "--fold", "0",
             "--epochs", "50", "--batch-size", "16"],
            capture_output=True, text=True, cwd=config.BASE_DIR,
            timeout=600,
        )
        lines = result.stdout.split("\n")

    # Parse epoch lines
    epochs, train_losses, val_losses, val_accs = [], [], [], []
    pattern = r"Epoch\s+(\d+)/\d+\s+\|\s+train_loss=([\d.]+)\s+\|\s+val_loss=([\d.]+)\s+\|\s+val_acc=([\d.]+)"

    for line in lines:
        match = re.search(pattern, line)
        if match:
            epochs.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            val_losses.append(float(match.group(3)))
            val_accs.append(float(match.group(4)))

    if not epochs:
        print("  [training curves] No training data found to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, train_losses, "b-o", markersize=3, label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-o", markersize=3, label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend()

    # Accuracy curve
    ax2.plot(epochs, val_accs, "g-o", markersize=3, label="Val Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy", fontweight="bold")
    ax2.axhline(y=0.68, color="gray", linestyle="--", alpha=0.7, label="Baseline F1 (0.68)")
    ax2.set_ylim(0, 1)
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 4. Attention Weight Visualization
# ─────────────────────────────────────────
def plot_attention_weights():
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "audio_branch_fold0_binary-stress.pt")
    if not os.path.exists(ckpt_path):
        print("  [attention] No checkpoint found. Train a model first.")
        return

    # Load model
    branch = AudioBranch(
        n_mels=config.N_MELS, max_frames=config.MAX_FRAMES,
        cnn_channels=config.CNN_CHANNELS, embed_dim=config.EMBED_DIM,
        n_heads=config.TRANSFORMER_HEADS, n_layers=config.TRANSFORMER_LAYERS,
        ff_dim=config.TRANSFORMER_FF_DIM, dropout=config.TRANSFORMER_DROPOUT,
    )
    model = AudioClassifier(branch, num_classes=2)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()

    # Get samples
    labels_df = pd.read_csv(config.LABELS_CSV).set_index("subject/task")
    subjects = get_all_audio_subjects()

    examples = {"stressed": [], "no-stress": []}
    for subj in subjects:
        for task in config.AUDIO_TASKS:
            key = f"{subj}_{task}"
            npy_path = os.path.join(config.MEL_DIR, subj, f"{task}_mel.npy")
            if key not in labels_df.index or not os.path.exists(npy_path):
                continue
            label = int(labels_df.loc[key, "binary-stress"])
            cat = "stressed" if label == 1 else "no-stress"
            if len(examples[cat]) < 2:
                examples[cat].append((key, npy_path))
            if all(len(v) >= 2 for v in examples.values()):
                break

    all_examples = examples["no-stress"] + examples["stressed"]
    fig, axes = plt.subplots(len(all_examples), 2, figsize=(16, 3.5 * len(all_examples)))

    for i, (key, npy_path) in enumerate(all_examples):
        mel = np.load(npy_path)
        x = torch.from_numpy(mel).float().unsqueeze(0)

        with torch.no_grad():
            _, attn_weights = branch.forward_with_attention(x)
        attn = attn_weights.squeeze(0).numpy()  # (234,)

        label = "Stressed" if i >= 2 else "No Stress"
        color = "#F44336" if i >= 2 else "#4CAF50"

        # Mel spectrogram
        axes[i, 0].imshow(mel.squeeze(0), aspect="auto", origin="lower", cmap="magma")
        axes[i, 0].set_title(f"{key} [{label}]", fontsize=11, color=color, fontweight="bold")
        axes[i, 0].set_ylabel("Mel Bin")
        if i == len(all_examples) - 1:
            axes[i, 0].set_xlabel("Time Frame")

        # Attention weights
        axes[i, 1].bar(range(len(attn)), attn, color=color, alpha=0.8, width=1.0)
        axes[i, 1].set_title(f"Attention Weights [{label}]", fontsize=11, fontweight="bold")
        axes[i, 1].set_ylabel("Weight")
        axes[i, 1].set_xlim(0, len(attn))
        if i == len(all_examples) - 1:
            axes[i, 1].set_xlabel("Timestep (post-CNN)")

    fig.suptitle("Temporal Attention Pooling: Where the Model Focuses", fontweight="bold", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "attention_weights.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 5. Architecture Diagram (text-based)
# ─────────────────────────────────────────
def plot_architecture():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # Each block: (x, y_center, text, facecolor, width, height)
    blocks = [
        (5, 13.0, "Input: Mel Spectrogram\n(B, 1, 128, 1876)", "#E3F2FD", 3.5, 1.1),
        (5, 11.5, "2D-CNN Backbone\n3x [Conv2d + BN + ReLU + MaxPool2x2]\n(B, 128, 16, 234)", "#BBDEFB", 3.5, 1.1),
        (5, 9.8, "Frequency Collapse\nAverage over mel axis\n(B, 128, 234) -> (B, 234, 128)", "#90CAF9", 3.5, 1.1),
        (5, 8.2, "Sinusoidal Positional Encoding\n(B, 234, 128)", "#64B5F6", 3.5, 1.1),
        (5, 6.5, "Transformer Encoder\n2 layers, 4 heads, d=128, ff=256\n(B, 234, 128)", "#42A5F5", 3.5, 1.1),
        (5, 4.8, "Attention Pooling\nLearnable query + softmax weighting\n(B, 128)", "#1E88E5", 3.5, 1.1),
        # Taller box (height=1.5) so the 3-line text fits comfortably
        (5, 3.1, "Classification Head\nLinear(128,64) + ReLU\n+ Dropout(0.3) + Linear(64,2)", "#1565C0", 3.5, 1.5),
        (5, 1.6, "Output: Stress Prediction\n(B, 2)", "#0D47A1", 3.5, 1.1),
    ]

    for x, y, text, color, width, height in blocks:
        half_h = height / 2
        box = plt.Rectangle((x - width / 2, y - half_h), width, height,
                            facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, fontweight="bold")

    # Arrows: connect bottom of block[i] to top of block[i+1]
    for i in range(len(blocks) - 1):
        y_start = blocks[i][1] - blocks[i][5] / 2  # bottom of current block
        y_end = blocks[i + 1][1] + blocks[i + 1][5] / 2  # top of next block
        ax.annotate("", xy=(5, y_end), xytext=(5, y_start),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#333"))

    # Side annotation for fusion (points to Attention Pooling block at y=4.8)
    ax.annotate("128-d embedding\nfor fusion", xy=(6.8, 4.8), fontsize=10, color="#E65100",
                fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#E65100"),
                xytext=(7.5, 4.1))

    ax.set_title("Audio Branch Architecture\nCNN + Transformer + Attention Pooling",
                 fontweight="bold", fontsize=14, pad=10)

    path = os.path.join(OUT_DIR, "architecture_diagram.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 6. Data Split Visualization
# ─────────────────────────────────────────
def plot_splits():
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = {"Train": "#4CAF50", "Val": "#FFC107", "Test": "#F44336"}

    for fold in range(config.NUM_FOLDS):
        train_s, val_s, test_s = get_subject_splits(fold=fold)
        train_ds = StressAudioDataset(train_s, label_col="binary-stress")
        val_ds = StressAudioDataset(val_s, label_col="binary-stress")
        test_ds = StressAudioDataset(test_s, label_col="binary-stress")

        y = config.NUM_FOLDS - 1 - fold
        left = 0
        for name, count in [("Train", len(train_ds)), ("Val", len(val_ds)), ("Test", len(test_ds))]:
            bar = ax.barh(y, count, left=left, color=colors[name], edgecolor="black",
                          linewidth=0.5, height=0.6, label=name if fold == 0 else "")
            if count > 20:
                ax.text(left + count / 2, y, f"{count}", ha="center", va="center",
                        fontweight="bold", fontsize=10)
            left += count

    ax.set_yticks(range(config.NUM_FOLDS))
    ax.set_yticklabels([f"Fold {config.NUM_FOLDS - 1 - i}" for i in range(config.NUM_FOLDS)])
    ax.set_xlabel("Number of Samples")
    ax.set_title("Subject-Level 5-Fold Cross-Validation Splits", fontweight="bold")
    ax.legend(loc="lower right")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cv_splits.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        choices=["dataset", "spectrogram", "training", "attention", "architecture", "splits"])
    parser.add_argument("--log", type=str, default=None, help="Training log file for curves")
    args = parser.parse_args()

    print(f"[visualize] Saving figures to: {OUT_DIR}")

    if args.only is None or args.only == "dataset":
        print("\n--- Dataset Statistics ---")
        plot_dataset_stats()

    if args.only is None or args.only == "spectrogram":
        print("\n--- Mel Spectrogram Examples ---")
        plot_spectrograms()

    if args.only is None or args.only == "architecture":
        print("\n--- Architecture Diagram ---")
        plot_architecture()

    if args.only is None or args.only == "splits":
        print("\n--- CV Split Visualization ---")
        plot_splits()

    if args.only is None or args.only == "attention":
        print("\n--- Attention Weight Visualization ---")
        plot_attention_weights()

    if args.only == "training":
        print("\n--- Training Curves ---")
        plot_training_curves(log_file=args.log)

    print(f"\n[visualize] Done! All figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
