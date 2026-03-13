import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from stress_dataset import (
    compute_metrics, run_kfold, evaluate_on_heldout,
    print_kfold_summary, print_heldout_summary,
    MAX_FRAMES, N_LANDMARKS, N_COORDS,
    FACE_DIR, CSV_PATH, TEST_IDS,
)

INPUT_SIZE = N_LANDMARKS * N_COORDS
CNN_CHANNELS = [256, 128]
KERNEL_SIZE = 3
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


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
    import os
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"cnn_lstm_fold{fold}.pt")

    model = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_metrics = {"RMSE": float("inf"),
                    "Pearson": 0.0, "MSE": float("inf")}
    best_state = None
    history = {"loss": [], "rmse": [], "pearson": []}

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

        history["loss"].append(train_loss)
        history["rmse"].append(metrics["RMSE"])
        history["pearson"].append(metrics["Pearson"])

        print(f"  {epoch:>6}  {train_loss:>10.4f}  "
              f"{metrics['RMSE']:>8.4f}  {metrics['Pearson']:>8.4f}")

        if metrics["RMSE"] < best_metrics["RMSE"]:
            best_metrics = metrics
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)

    model.load_state_dict(best_state)
    return best_metrics, model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--face_dir", default=FACE_DIR)
    parser.add_argument("--csv_path", default=CSV_PATH)
    parser.add_argument("--test_ids", nargs="+", default=TEST_IDS)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--save_dir", default="checkpoints")
    args = parser.parse_args()

    fold_results, held_out_samples, fold_models = run_kfold(
        model_fn=StressCNNLSTM,
        train_fn=train_one_fold,
        face_dir=args.face_dir,
        csv_path=args.csv_path,
        test_ids=args.test_ids,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        model_name="CNN-LSTM",
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
    )

    print_kfold_summary({"CNN-LSTM": fold_results})

    best_idx = min(range(len(fold_results)), key=lambda i: fold_results[i]["RMSE"])
    best_model = fold_models[best_idx]
    heldout_metrics = evaluate_on_heldout(best_model, held_out_samples,
                                          model_name="CNN-LSTM", device=DEVICE)
    print_heldout_summary({"CNN-LSTM": heldout_metrics})
