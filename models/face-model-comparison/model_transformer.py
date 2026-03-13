import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from stress_dataset import (
    compute_metrics, run_kfold, evaluate_on_heldout,
    print_kfold_summary, print_heldout_summary,
    MAX_FRAMES, N_LANDMARKS, N_COORDS,
    FACE_DIR, CSV_PATH, TEST_IDS,
)

INPUT_SIZE = N_LANDMARKS * N_COORDS
D_MODEL    = 256
N_HEADS    = 8
N_LAYERS   = 3
DIM_FF     = 512
DROPOUT    = 0.1
BATCH_SIZE = 16
EPOCHS     = 10
LR         = 1e-4
DEVICE     = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_FRAMES + 1, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class StressTransformer(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, dim_ff=DIM_FF, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 128),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        B  = x.size(0)
        x  = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x  = torch.cat([cls, x], dim=1)
        x  = self.pos_enc(x)
        x  = self.encoder(x)
        return self.head(x[:, 0, :]).squeeze(-1)


def train_one_fold(model, train_loader, val_loader,
                   fold=0, epochs=EPOCHS, lr=LR, save_dir="checkpoints"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"transformer_fold{fold}.pt")

    model     = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_metrics = {"RMSE": float("inf"),
                    "Pearson": 0.0, "MSE": float("inf")}
    best_state   = None
    history      = {"loss": [], "rmse": [], "pearson": []}

    print(f"\n  {'Epoch':>6}  {'TrainLoss':>10}  {'RMSE':>8}  {'Pearson':>8}")
    print(f"  {'-'*46}")

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
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)

    model.load_state_dict(best_state)
    return best_metrics, model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--face_dir",   default=FACE_DIR)
    parser.add_argument("--csv_path",   default=CSV_PATH)
    parser.add_argument("--test_ids",   nargs="+", default=TEST_IDS)
    parser.add_argument("--n_splits",   type=int,   default=5)
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--save_dir",   default="checkpoints")
    args = parser.parse_args()

    fold_results, held_out_samples, fold_models = run_kfold(
        model_fn    = StressTransformer,
        train_fn    = train_one_fold,
        face_dir    = args.face_dir,
        csv_path    = args.csv_path,
        test_ids    = args.test_ids,
        n_splits    = args.n_splits,
        batch_size  = args.batch_size,
        model_name  = "Transformer",
        epochs      = args.epochs,
        lr          = args.lr,
        save_dir    = args.save_dir,
    )

    print_kfold_summary({"Transformer": fold_results})

    best_idx   = min(range(len(fold_results)), key=lambda i: fold_results[i]["RMSE"])
    best_model = fold_models[best_idx]
    heldout_metrics = evaluate_on_heldout(best_model, held_out_samples,
                                          model_name="Transformer", device=DEVICE)
    print_heldout_summary({"Transformer": heldout_metrics})