import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              balanced_accuracy_score, matthews_corrcoef,
                              classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

import config
from models.audio_branch import AudioBranch, AudioClassifier
from models.face_branch import FaceBranch
from models.gesture_branch import GestureBranch, GestureClassifier
from models.fusion import LateFusionClassifier, CrossAttentionFusionClassifier
from models.gated_fusion import GatedFusionClassifier
from dataset_multimodal import (MultimodalStressDataset,
                                 get_multimodal_subject_splits,
                                 get_multimodal_held_out_subjects)


def compute_metrics(preds, labels, probs=None, num_classes=2):
    acc = accuracy_score(labels, preds)
    f1w = f1_score(labels, preds, average="weighted", zero_division=0)
    f1m = f1_score(labels, preds, average="macro", zero_division=0)
    bal = balanced_accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    auc = float("nan")
    if probs is not None and len(np.unique(labels)) > 1:
        try:
            if num_classes == 2:
                auc = roc_auc_score(labels, probs[:, 1])
            else:
                auc = roc_auc_score(labels, probs, multi_class="ovr")
        except Exception:
            pass
    return {"acc": acc, "f1w": f1w, "f1m": f1m, "bal_acc": bal, "mcc": mcc, "auc": auc}


def load_audio_branch(ckpt_path, device):
    branch = AudioBranch()
    classifier = AudioClassifier(branch, num_classes=2, hidden_dim=128)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    classifier.load_state_dict(state)
    return classifier.backbone


def load_gesture_branch(ckpt_path, device):
    branch = GestureBranch(
        input_size=config.GESTURE_CKPT_INPUT_DIM,
        joint_count=config.UPPER_BODY_N_LANDMARKS,
        coord_dim=3,
    )
    classifier = GestureClassifier(branch, num_classes=2, hidden_dim=64)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    classifier.load_state_dict(state)
    return classifier.backbone


def build_model(fusion_type, audio_branch, num_classes, device, gesture_branch=None):
    face_branch = FaceBranch(input_dim=286)

    if fusion_type == "late":
        model = LateFusionClassifier(
            audio_branch=audio_branch,
            face_branch=face_branch,
            gesture_branch=gesture_branch,
            num_classes=num_classes,
            hidden_dim=128,
            dropout=0.3,
        )
    elif fusion_type == "gated":
        model = GatedFusionClassifier(
            audio_branch, face_branch, gesture_branch,
            num_classes=num_classes, hdim=128, drop=0.3,
        )
    else:
        model = CrossAttentionFusionClassifier(
            audio_branch=audio_branch,
            face_branch=face_branch,
            gesture_branch=gesture_branch,
            embed_dim=config.EMBED_DIM,
            n_heads=config.TRANSFORMER_HEADS,
            num_classes=num_classes,
            dropout=config.TRANSFORMER_DROPOUT,
            classifier_dropout=0.3,
        )

    return model.to(device)


def _forward(model, batch, device):
    if len(batch) == 4:
        mel, face, gesture, labels = batch
        mel, face, gesture, labels = (mel.to(device), face.to(device),
                                      gesture.to(device), labels.to(device))
        logits = model(mel, face, gesture_input=gesture)
    else:
        mel, face, labels = batch
        mel, face, labels = mel.to(device), face.to(device), labels.to(device)
        logits = model(mel, face)
    return logits, labels


def train_one_fold(fold, fusion_type, label_col, num_classes, args, device):
    print(f"\n{'='*65}")
    print(f"  Fold {fold + 1}/{config.NUM_FOLDS}  |  fusion={fusion_type}  |  label={label_col}")
    print(f"{'='*65}")

    train_subj, val_subj, test_subj = get_multimodal_subject_splits(
        fold=fold, label_col=label_col)

    print(f"  Loading data (mel extraction may take a moment)...")
    t0 = time.time()
    train_ds = MultimodalStressDataset(train_subj, label_col=label_col, preload=True)
    val_ds   = MultimodalStressDataset(val_subj,   label_col=label_col, preload=True)
    test_ds  = MultimodalStressDataset(test_subj,  label_col=label_col, preload=True)
    print(f"  Data loaded in {time.time()-t0:.1f}s  "
          f"| train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    if len(train_ds) == 0:
        print("  [WARN] Empty train set — skipping fold.")
        return None, None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                               num_workers=0)

    all_labels = [s[3] for s in train_ds.samples]
    cw = compute_class_weight("balanced", classes=np.arange(num_classes),
                               y=all_labels)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float).to(device))

    audio_branch = load_audio_branch(config.AUDIO_CKPT_PATH, device)
    gesture_branch = None
    if train_ds.use_gesture and os.path.exists(config.GESTURE_CKPT_PATH):
        gesture_branch = load_gesture_branch(config.GESTURE_CKPT_PATH, device)
    model = build_model(fusion_type, audio_branch, num_classes, device,
                        gesture_branch=gesture_branch)

    if args.unfreeze_audio:
        model.unfreeze_audio()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,} trainable / {total:,} total")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    best_f1m = -1.0
    best_metrics = None
    best_state = None
    no_improve = 0

    print(f"\n  {'Ep':>4}  {'TrLoss':>8}  {'ValAcc':>8}  {'ValF1m':>8}  "
          f"{'ValAUC':>8}  {'ValMCC':>7}")
    print(f"  {'-'*55}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, labels = _forward(model, batch, device)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item() * labels.size(0)
        tr_loss /= len(train_ds)

        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                logits, labels = _forward(model, batch, device)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        m = compute_metrics(np.array(all_preds), np.array(all_labels),
                            np.array(all_probs), num_classes)
        scheduler.step(m["f1m"])

        print(f"  {epoch:>4}  {tr_loss:>8.4f}  {m['acc']:>8.4f}  {m['f1m']:>8.4f}  "
              f"{m['auc']:>8.4f}  {m['mcc']:>7.4f}")

        if m["f1m"] > best_f1m:
            best_f1m = m["f1m"]
            best_metrics = m.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  [EarlyStop] no improvement for {args.patience} epochs")
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            logits, labels = _forward(model, batch, device)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    test_m = compute_metrics(np.array(all_preds), np.array(all_labels),
                              np.array(all_probs), num_classes)
    print(f"\n  +- Fold {fold+1} Val Best  -> f1m={best_f1m:.4f}")
    print(f"  +- Fold {fold+1} Test      "
          f"acc={test_m['acc']:.4f}  f1m={test_m['f1m']:.4f}  "
          f"auc={test_m['auc']:.4f}  mcc={test_m['mcc']:.4f}")

    return best_metrics, best_state, test_m


def eval_held_out(best_state, fusion_type, label_col, num_classes, device):
    held_out = get_multimodal_held_out_subjects()
    if not held_out:
        print("\n  [held-out] No held-out subjects with face features found.")
        return {}

    print(f"\n{'='*65}")
    print(f"  HELD-OUT EVAL  subjects={held_out}")
    print(f"{'='*65}")

    ds = MultimodalStressDataset(held_out, label_col=label_col, preload=True)
    if len(ds) == 0:
        print("  [held-out] No matching samples found.")
        return {}

    loader = DataLoader(ds, batch_size=16, shuffle=False)

    audio_branch = load_audio_branch(config.AUDIO_CKPT_PATH, device)
    gesture_branch = None
    if ds.use_gesture and os.path.exists(config.GESTURE_CKPT_PATH):
        gesture_branch = load_gesture_branch(config.GESTURE_CKPT_PATH, device)
    model = build_model(fusion_type, audio_branch, num_classes, device,
                        gesture_branch=gesture_branch)
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits, labels = _forward(model, batch, device)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    m = compute_metrics(np.array(all_preds), np.array(all_labels),
                        np.array(all_probs), num_classes)
    target_names = ["no-stress", "stressed"] if num_classes == 2 else ["c0", "c1", "c2"]
    print(classification_report(all_labels, all_preds,
                                 target_names=target_names, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print(f"\n  acc={m['acc']:.4f}  f1m={m['f1m']:.4f}  "
          f"auc={m['auc']:.4f}  mcc={m['mcc']:.4f}")
    return m


def main(args):
    device = config.DEVICE
    print(f"Device: {device}")
    print(f"Fusion: {args.fusion}  |  Label: {args.label}")
    print(f"Audio ckpt: {config.AUDIO_CKPT_PATH}")
    print(f"Face dir  : {config.FACE_FEATURE_DIR}")

    num_classes = config.NUM_CLASSES_BINARY if args.label == "binary-stress" else config.NUM_CLASSES_AFFECT3

    fold_val_results = []
    fold_test_results = []
    fold_states = []

    for fold in range(config.NUM_FOLDS):
        result = train_one_fold(fold, args.fusion, args.label, num_classes, args, device)
        if result[0] is None:
            continue
        val_m, state, test_m = result
        fold_val_results.append(val_m)
        fold_test_results.append(test_m)
        fold_states.append(state)

    if not fold_val_results:
        print("No folds completed.")
        return

    keys = ["acc", "f1w", "f1m", "auc", "bal_acc", "mcc"]
    print(f"\n{'='*65}")
    print(f"  5-FOLD CV SUMMARY  fusion={args.fusion}  label={args.label}")
    print(f"{'='*65}")
    print(f"  {'Metric':<14}  {'Val Mean':>10}  {'Val Std':>8}  "
          f"{'Test Mean':>10}  {'Test Std':>8}")
    for k in keys:
        vv = [r[k] for r in fold_val_results if not np.isnan(r[k])]
        tv = [r[k] for r in fold_test_results if not np.isnan(r[k])]
        print(f"  {k:<14}  {np.mean(vv):>10.4f}  {np.std(vv):>8.4f}  "
              f"{np.mean(tv):>10.4f}  {np.std(tv):>8.4f}")
    print(f"{'='*65}")

    best_fold = max(range(len(fold_val_results)),
                    key=lambda i: fold_val_results[i]["f1m"])
    print(f"  Best fold: {best_fold + 1}  "
          f"(val f1m={fold_val_results[best_fold]['f1m']:.4f})")

    heldout_m = eval_held_out(fold_states[best_fold], args.fusion,
                               args.label, num_classes, device)

    results = {
        "fusion": args.fusion,
        "label": args.label,
        "fold_val": fold_val_results,
        "fold_test": fold_test_results,
        "best_fold": best_fold,
        "held_out": heldout_m,
    }
    out_path = os.path.join(
        config.BASE_DIR,
        f"fusion_results_{args.fusion}_{args.label}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {out_path}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(
        config.CHECKPOINT_DIR,
        f"fusion_{args.fusion}_{args.label}_best.pt")
    torch.save(fold_states[best_fold], ckpt_path)
    print(f"  Checkpoint  -> {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", choices=["late", "cross_attn", "gated"], default="cross_attn")
    parser.add_argument("--label", choices=["binary-stress", "affect3-class"],
                        default="binary-stress")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--unfreeze_audio", action="store_true")
    args = parser.parse_args()
    main(args)
