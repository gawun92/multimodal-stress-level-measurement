import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from stress_dataset import (
    build_subject_samples, StressDataset,
    evaluate_on_heldout, print_heldout_summary,
    FACE_DIR, CSV_PATH, TEST_IDS,
)
from model_lstm import StressLSTM
from model_transformer import StressTransformer
from model_cnn_lstm import StressCNNLSTM

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
COLORS = {"LSTM": "#1f77b4", "Transformer": "#ff7f0e", "CNN-LSTM": "#2ca02c"}


def train_model(model_cls, model_name, all_samples, held_out_samples, args):
    train_loader = DataLoader(StressDataset(all_samples),
                              batch_size=args.batch_size, shuffle=True)
    heldout_loader = DataLoader(
        StressDataset([(arr, score) for arr, score, *_ in held_out_samples]),
        batch_size=args.batch_size, shuffle=False)

    print(f"\n{'=' * 60}")
    print(f"  [{model_name}]  Full Training (Held-out Evaluation)")
    print(f"{'=' * 60}")
    print(f"  train : {len(all_samples)} samples  |  held-out : {len(held_out_samples)} samples")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{model_name.lower().replace('-', '_')}_full.pt")

    model = model_cls().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    history = {"loss": [], "rmse": []}

    print(f"\n  {'Epoch':>6}  {'TrainLoss':>10}  {'HeldRMSE':>10}")
    print(f"  {'-' * 32}")

    for epoch in range(1, args.epochs + 1):
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

        # held-out RMSE per epoch
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in heldout_loader:
                preds.extend(model(x.to(DEVICE)).cpu().numpy().tolist())
                targets.extend(y.numpy().tolist())
        rmse = float(np.sqrt(np.mean((np.array(preds) - np.array(targets)) ** 2)))

        history["loss"].append(train_loss)
        history["rmse"].append(rmse)
        print(f"  {epoch:>6}  {train_loss:>10.4f}  {rmse:>10.4f}")
        scheduler.step(train_loss)

    torch.save(model.state_dict(), save_path)
    print(f"\n  Model saved → {save_path}")
    return model, history


def save_training_loss(histories, epochs, save_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, h in histories.items():
        ax.plot(range(1, epochs + 1), h["loss"], label=name, color=COLORS[name])
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(save_dir, "plot_training_loss.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_rmse(histories, epochs, save_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, h in histories.items():
        ax.plot(range(1, epochs + 1), h["rmse"], label=name, color=COLORS[name])
    ax.set_title("RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(save_dir, "plot_rmse.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_heldout_bar(heldout_summary, save_dir):
    models = list(heldout_summary.keys())
    x = np.arange(len(models))
    width = 0.35
    colors = [COLORS[m] for m in models]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.bar(x - width / 2, [heldout_summary[m]["RMSE"] for m in models],
            width, color=colors, alpha=0.85)
    ax2.bar(x + width / 2, [heldout_summary[m]["Pearson"] for m in models],
            width, color=colors, alpha=0.45, hatch="//")

    ax1.set_title("Held-out Final Test")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylabel("RMSE ↓")
    ax2.set_ylabel("Pearson ↑")
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax1.grid(True, axis="y")

    legend_elements = [
        mpatches.Patch(facecolor="gray", alpha=0.85, label="RMSE (solid)"),
        mpatches.Patch(facecolor="gray", alpha=0.45, hatch="//", label="Pearson (hatch)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=8)

    best = min(models, key=lambda m: heldout_summary[m]["RMSE"])
    best_idx = models.index(best)
    ax1.annotate("★ Best", xy=(best_idx - width / 2, heldout_summary[best]["RMSE"]),
                 xytext=(0, 6), textcoords="offset points",
                 ha="center", fontsize=9, color="green", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(save_dir, "plot_heldout.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main(args):
    heldout_summary = {}
    histories = {}

    kfold_subjects, held_out_samples = build_subject_samples(
        face_dir=args.face_dir, csv_path=args.csv_path, test_ids=args.test_ids)

    all_samples = []
    for samples in kfold_subjects.values():
        all_samples.extend(samples)

    models_config = [
        ("LSTM", StressLSTM),
        ("Transformer", StressTransformer),
        ("CNN-LSTM", StressCNNLSTM),
    ]

    for model_name, model_cls in models_config:
        print(f"\n{'★' * 60}")
        print(f"  Running: {model_name}")
        print(f"{'★' * 60}")

        model, history = train_model(model_cls, model_name, all_samples, held_out_samples, args)
        histories[model_name] = history

        heldout_metrics = evaluate_on_heldout(
            model, held_out_samples,
            model_name=model_name, device=DEVICE,
        )
        heldout_summary[model_name] = heldout_metrics

    print_heldout_summary(heldout_summary)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"\n  Saving graphs to '{args.save_dir}/'")
    save_training_loss(histories, args.epochs, args.save_dir)
    save_rmse(histories, args.epochs, args.save_dir)
    save_heldout_bar(heldout_summary, args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--face_dir", default=FACE_DIR)
    parser.add_argument("--csv_path", default=CSV_PATH)
    parser.add_argument("--test_ids", nargs="+", default=TEST_IDS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", default="checkpoints")
    args = parser.parse_args()

    main(args)
