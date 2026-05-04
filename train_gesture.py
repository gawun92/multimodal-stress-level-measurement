"""
train_gesture.py

Train the gesture classifier on StressID gesture features.

The canonical gesture modality in this repo is upper-body/head motion.

Example usage:
    python train_gesture.py --fold 0 --label binary-stress
    python train_gesture.py --fold 0 --label affect3-class --linear-probe
    
    python train_gesture.py \
        --gesture-dir feature_extraction/results/upper_body/train_20260416_193709 \
        --label binary-stress \
        --fold 0 \
        --n-folds 3 \
        --require-mask \
        --window-len 96 \
        --min-valid-frames 24
"""

import argparse
from datetime import datetime
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

import config
from dataset import StressGestureDataset, get_all_gesture_subjects, get_subject_splits
from models.gesture_branch import GestureBranch, GestureClassifier


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
    parser = argparse.ArgumentParser(description="Train gesture branch for stress detection")
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
        default="affect3-class",
        choices=["binary-stress", "affect3-class"],
        help="Label column to train on",
    )
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument(
        "--linear-probe",
        action="store_true",
        help="Freeze backbone, train only classification head",
    )
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
        help="Require *_gesture_mask.npy files and use them during training",
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
    parser.add_argument(
        "--pretrained-backbone",
        type=str,
        default=None,
        help="Optional path to pretrained gesture backbone weights",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=config.CHECKPOINT_DIR)
    return parser.parse_args()


def get_class_weights(dataset, num_classes, device):
    labels = np.array([sample[2] for sample in dataset.samples], dtype=np.int64)
    weights = np.ones(num_classes, dtype=np.float32)
    present_classes = np.unique(labels)
    if present_classes.size == 0:
        return torch.tensor(weights, dtype=torch.float32).to(device)

    balanced = compute_class_weight("balanced", classes=present_classes, y=labels)
    weights[present_classes] = balanced.astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def apply_gesture_mask(X, mask):
    return X * mask.unsqueeze(-1).to(dtype=X.dtype)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_valid = 0.0

    for batch in dataloader:
        if len(batch) == 3:
            X, mask, y = batch
            X = apply_gesture_mask(X, mask)
            total_valid += mask.float().mean().item()
        else:
            mask = None
            X, y = batch
        X, y = X.to(device), y.to(device)
        if mask is not None:
            mask = mask.to(device)
        optimizer.zero_grad()
        logits = model(X, mask=mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    avg_valid = total_valid / max(n_batches, 1) if total_valid > 0 else None
    return total_loss / max(n_batches, 1), avg_valid


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0
    total_valid = 0.0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                X, mask, y = batch
                X = apply_gesture_mask(X, mask)
                total_valid += mask.float().mean().item()
            else:
                mask = None
                X, y = batch
            X, y = X.to(device), y.to(device)
            if mask is not None:
                mask = mask.to(device)
            logits = model(X, mask=mask)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = correct / max(total, 1)
    avg_valid = total_valid / max(n_batches, 1) if total_valid > 0 else None
    return avg_loss, accuracy, avg_valid


def main():
    args = parse_args()
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    label_col = args.label
    num_classes = (
        config.NUM_CLASSES_BINARY
        if label_col == "binary-stress"
        else config.NUM_CLASSES_AFFECT3
    )
    gesture_tasks = list(config.VIDEO_TASKS)
    if not args.include_baseline:
        gesture_tasks = [task for task in gesture_tasks if task != "Baseline"]

    train_subjects, val_subjects, test_subjects = get_subject_splits(
        fold=args.fold,
        n_folds=args.n_folds,
        subject_fn=lambda: get_all_gesture_subjects(args.gesture_dir),
        tasks=gesture_tasks,
    )
    print(
        f"[train_gesture] Fold {args.fold}: train={len(train_subjects)}, "
        f"val={len(val_subjects)}, test={len(test_subjects)} subjects"
    )
    print(f"[train_gesture] Gesture dir: {args.gesture_dir}")
    print(f"[train_gesture] Tasks: {gesture_tasks}")

    joint_count, coord_dim, input_dim, schema_path = detect_gesture_schema(args.gesture_dir)
    if (
        joint_count != config.GESTURE_N_LANDMARKS
        or input_dim != config.GESTURE_INPUT_DIM
    ):
        print(
            "[train_gesture] Detected gesture schema override: "
            f"{joint_count} landmarks x {coord_dim} coords = {input_dim} dims "
            f"(from {schema_path})"
        )
    else:
        print(
            f"[train_gesture] Detected gesture schema: "
            f"{joint_count} landmarks x {coord_dim} coords"
        )

    train_ds = StressGestureDataset(
        train_subjects,
        label_col=label_col,
        gesture_dir=args.gesture_dir,
        tasks=gesture_tasks,
        return_mask=args.require_mask,
        require_mask=args.require_mask,
        window_len=args.window_len,
        min_valid_frames=args.min_valid_frames,
    )
    val_ds = StressGestureDataset(
        val_subjects,
        label_col=label_col,
        gesture_dir=args.gesture_dir,
        tasks=gesture_tasks,
        return_mask=args.require_mask,
        require_mask=args.require_mask,
        window_len=args.window_len,
        min_valid_frames=args.min_valid_frames,
    )
    print(f"[train_gesture] Samples: train={len(train_ds)}, val={len(val_ds)}")
    print(f"[train_gesture] Mask mode: {'ON' if args.require_mask else 'OFF'}")

    if len(train_ds) == 0:
        print(
            "[train_gesture] ERROR: No training samples found. "
            "Run gesture feature extraction first and/or point --gesture-dir to the extracted subset."
        )
        return

    pin_memory = config.DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    branch = GestureBranch(
        input_size=input_dim,
        embed_dim=config.EMBED_DIM,
        joint_count=joint_count,
        coord_dim=coord_dim,
    )
    if args.pretrained_backbone:
        if not os.path.exists(args.pretrained_backbone):
            raise FileNotFoundError(
                f"Pretrained backbone checkpoint not found: {args.pretrained_backbone}"
            )
        backbone_state = torch.load(
            args.pretrained_backbone,
            map_location=config.DEVICE,
            weights_only=True,
        )
        branch.load_state_dict(backbone_state)
        print(f"[train_gesture] Loaded pretrained backbone: {args.pretrained_backbone}")

    model = GestureClassifier(branch, num_classes=num_classes)
    if args.linear_probe:
        model.freeze_backbone()
        print("[train_gesture] Linear probing mode: backbone frozen")

    model = model.to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[train_gesture] Parameters: {total_params:,} total, "
        f"{trainable_params:,} trainable"
    )

    class_weights = get_class_weights(train_ds, num_classes, config.DEVICE)
    print(f"[train_gesture] Class weights: {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.checkpoint_dir,
        f"gesture_branch_fold{args.fold}_{label_col}_{run_ts}.pt",
    )
    log_path = os.path.join(
        args.checkpoint_dir,
        f"gesture_train_fold{args.fold}_{label_col}_{run_ts}.txt",
    )

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("Gesture Training Log\n")
        log_file.write(f"Timestamp: {run_ts}\n")
        log_file.write(f"Fold: {args.fold}\n")
        log_file.write(f"N-Folds: {args.n_folds}\n")
        log_file.write(f"Label: {label_col}\n")
        log_file.write(f"Gesture dir: {args.gesture_dir}\n")
        log_file.write(f"Mask mode: {'ON' if args.require_mask else 'OFF'}\n")
        log_file.write(
            f"Window len: {args.window_len if args.window_len is not None else 'full'}\n"
        )
        log_file.write(f"Min valid frames: {args.min_valid_frames}\n")
        log_file.write(f"Linear probe: {'ON' if args.linear_probe else 'OFF'}\n")
        log_file.write(f"Pretrained backbone: {args.pretrained_backbone or 'None'}\n")
        log_file.write(f"Train subjects: {len(train_subjects)}\n")
        log_file.write(f"Val subjects: {len(val_subjects)}\n")
        log_file.write(f"Test subjects: {len(test_subjects)}\n")
        log_file.write(f"Train samples: {len(train_ds)}\n")
        log_file.write(f"Val samples: {len(val_ds)}\n")
        log_file.write(f"Tasks: {gesture_tasks}\n")
        log_file.write(
            "epoch\ttrain_loss\tval_loss\tval_acc\ttrain_valid\tval_valid\tlr\tbest\n"
        )

        for epoch in range(args.epochs):
            train_loss, train_valid = train_one_epoch(
                model, train_loader, criterion, optimizer, config.DEVICE
            )
            val_loss, val_acc, val_valid = validate(
                model, val_loader, criterion, config.DEVICE
            )
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            valid_stats = ""
            if args.require_mask:
                train_valid_str = (
                    f"{train_valid:.3f}" if train_valid is not None else "n/a"
                )
                val_valid_str = f"{val_valid:.3f}" if val_valid is not None else "n/a"
                valid_stats = (
                    f" | train_valid={train_valid_str} | val_valid={val_valid_str}"
                )
            print(
                f"  Epoch {epoch + 1:3d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f}{valid_stats} | lr={current_lr:.2e}"
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1

            log_file.write(
                f"{epoch + 1}\t"
                f"{train_loss:.6f}\t"
                f"{val_loss:.6f}\t"
                f"{val_acc:.6f}\t"
                f"{train_valid if train_valid is not None else 'n/a'}\t"
                f"{val_valid if val_valid is not None else 'n/a'}\t"
                f"{current_lr:.8e}\t"
                f"{int(is_best)}\n"
            )
            log_file.flush()

            if not is_best and patience_counter >= config.PATIENCE:
                print(f"[train_gesture] Early stopping at epoch {epoch + 1}")
                log_file.write(f"Early stopping epoch: {epoch + 1}\n")
                break

        log_file.write(f"Best val_loss: {best_val_loss:.6f}\n")
        log_file.write(f"Checkpoint: {ckpt_path}\n")

    print(f"[train_gesture] Best val_loss: {best_val_loss:.4f}")
    print(f"[train_gesture] Checkpoint saved: {ckpt_path}")
    print(f"[train_gesture] Training log: {log_path}")


if __name__ == "__main__":
    main()
