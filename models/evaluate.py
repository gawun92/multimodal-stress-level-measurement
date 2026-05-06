"""
evaluate.py

Evaluation script for the audio branch.
Loads a trained checkpoint and computes metrics on the test fold.

Usage:
    python evaluate.py --label binary-stress --fold 0
    python evaluate.py --label affect3-class --fold 0
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import config
from dataset import StressAudioDataset, get_subject_splits
from models.audio_branch import AudioBranch, AudioClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate audio branch")
    parser.add_argument("--label", type=str, default="binary-stress",
                        choices=["binary-stress", "affect3-class"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--checkpoint-dir", type=str, default=config.CHECKPOINT_DIR)
    return parser.parse_args()


def evaluate(model, dataloader, device):
    """Run inference, return (all_preds, all_labels, all_probs) as numpy arrays."""
    import torch.nn.functional as F
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


def main():
    args = parse_args()

    num_classes = (config.NUM_CLASSES_BINARY if args.label == "binary-stress"
                   else config.NUM_CLASSES_AFFECT3)

    # Get test subjects for this fold
    _, _, test_subjects = get_subject_splits(fold=args.fold)
    test_ds = StressAudioDataset(test_subjects, label_col=args.label)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    print(f"[evaluate] Fold {args.fold}: {len(test_subjects)} test subjects, "
          f"{len(test_ds)} samples")

    if len(test_ds) == 0:
        print("[evaluate] ERROR: No test samples found.")
        return

    # Load model
    branch = AudioBranch(
        n_mels=config.N_MELS,
        max_frames=config.MAX_FRAMES,
        cnn_channels=config.CNN_CHANNELS,
        embed_dim=config.EMBED_DIM,
        n_heads=config.TRANSFORMER_HEADS,
        n_layers=config.TRANSFORMER_LAYERS,
        ff_dim=config.TRANSFORMER_FF_DIM,
        dropout=config.TRANSFORMER_DROPOUT,
    )
    model = AudioClassifier(branch, num_classes=num_classes)

    ckpt_path = os.path.join(
        args.checkpoint_dir, f"audio_branch_fold{args.fold}_{args.label}.pt"
    )
    if not os.path.exists(ckpt_path):
        print(f"[evaluate] ERROR: Checkpoint not found: {ckpt_path}")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE, weights_only=True))
    model = model.to(config.DEVICE)

    # Evaluate
    preds, labels, probs = evaluate(model, test_loader, config.DEVICE)

    # Metrics
    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_m = f1_score(labels, preds, average="macro", zero_division=0)

    if num_classes == 2:
        auc = roc_auc_score(labels, probs[:, 1])
    else:
        auc = roc_auc_score(labels, probs, multi_class="ovr")

    if args.label == "binary-stress":
        target_names = ["no-stress", "stressed"]
    else:
        target_names = ["class-0", "class-1", "class-2"]

    report = classification_report(labels, preds, target_names=target_names, zero_division=0)
    cm = confusion_matrix(labels, preds)

    print(f"\n{'=' * 50}")
    print(f"Results: fold={args.fold}, task={args.label}")
    print(f"{'=' * 50}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Weighted F1: {f1_w:.4f}")
    print(f"  Macro F1:    {f1_m:.4f}")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"\n{report}")
    print(f"Confusion Matrix:\n{cm}")


if __name__ == "__main__":
    main()
