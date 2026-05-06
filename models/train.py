"""
train.py

Standalone training loop for the audio branch.

Supports:
    - Subject-level k-fold cross-validation
    - Single-fold training with early stopping
    - Linear probing mode (frozen backbone)
    - Class-weighted loss for imbalanced labels

Usage:
    python train.py --label binary-stress --fold 0
    python train.py --label binary-stress --fold 0 --linear-probe
    python train.py --label affect3-class --fold 0
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

import config
from dataset import StressAudioDataset, get_subject_splits
from models.audio_branch import AudioBranch, AudioClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train audio branch for stress detection")
    parser.add_argument("--label", type=str, default="binary-stress",
                        choices=["binary-stress", "affect3-class"])
    parser.add_argument("--fold", type=int, default=0, help="CV fold index (0-based)")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--linear-probe", action="store_true",
                        help="Freeze backbone, train only classification head")
    parser.add_argument("--augment", action="store_true",
                        help="Apply SpecAugment during training")
    parser.add_argument("--checkpoint-dir", type=str, default=config.CHECKPOINT_DIR)
    return parser.parse_args()


def get_class_weights(dataset, num_classes, device):
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    labels = np.array([label for _, label in dataset.samples])
    classes = np.arange(num_classes)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, dataloader, criterion, device):
    """Validate, return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main():
    args = parse_args()

    num_classes = (config.NUM_CLASSES_BINARY if args.label == "binary-stress"
                   else config.NUM_CLASSES_AFFECT3)

    # Subject splits
    train_subjects, val_subjects, test_subjects = get_subject_splits(fold=args.fold)
    print(f"[train] Fold {args.fold}: train={len(train_subjects)}, "
          f"val={len(val_subjects)}, test={len(test_subjects)} subjects")

    # Datasets
    train_ds = StressAudioDataset(train_subjects, label_col=args.label, augment=args.augment)
    val_ds = StressAudioDataset(val_subjects, label_col=args.label, augment=False)
    print(f"[train] Samples: train={len(train_ds)}, val={len(val_ds)}")

    if len(train_ds) == 0:
        print("[train] ERROR: No training samples found. Run feature extraction first.")
        return

    # pin_memory not supported on MPS
    _pin = config.DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=_pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=_pin
    )

    # Model
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

    if args.linear_probe:
        model.freeze_backbone()
        print("[train] Linear probing mode: backbone frozen")
    print(f"[train] SpecAugment: {'ON' if args.augment else 'OFF'}")

    model = model.to(config.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Class-weighted loss
    class_weights = get_class_weights(train_ds, num_classes, config.DEVICE)
    print(f"[train] Class weights: {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer (only unfrozen params)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.checkpoint_dir, f"audio_branch_fold{args.fold}_{args.label}.pt"
    )

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch + 1:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"[train] Early stopping at epoch {epoch + 1}")
                break

    print(f"[train] Best val_loss: {best_val_loss:.4f}")
    print(f"[train] Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
