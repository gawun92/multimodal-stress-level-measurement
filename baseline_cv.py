import os
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import config

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(os.path.dirname(BASE_DIR), "Features")
DEVICE       = torch.device("mps" if torch.backends.mps.is_available()
                            else "cuda" if torch.cuda.is_available() else "cpu")

SEED    = 42
N_FOLDS = config.NUM_FOLDS
HELD_OUT = set(config.HELD_OUT_SUBJECTS)


def load_features():
    x_phys  = pd.read_csv(os.path.join(FEATURES_DIR, "all_physiological_features.csv"),
                           index_col=0)
    x_video = pd.read_csv(os.path.join(FEATURES_DIR, "video11tasks_aus_gaze_mean_std.csv"),
                           index_col=0)
    x_audio = pd.read_csv(os.path.join(FEATURES_DIR, "HCfeatures.csv"),
                           header=None, index_col=0)

    x_audio.index = x_audio.index.str.replace(r"\.wav$", "", regex=True)

    labels = pd.read_csv(config.LABELS_CSV, index_col=0)
    audio_tasks = set(config.AUDIO_TASKS)
    mask = [len(i.split("_", 1)) == 2 and i.split("_", 1)[1] in audio_tasks
            for i in labels.index]
    labels = labels[mask]

    X = (x_phys
         .merge(x_video, left_index=True, right_index=True, suffixes=("_p", "_v"))
         .merge(x_audio, left_index=True, right_index=True))
    X = X.merge(labels[["binary-stress"]], left_index=True, right_index=True).dropna()

    y = X.pop("binary-stress").astype(int)
    return X, y, len(x_phys.columns), len(x_video.columns)


def get_subject_splits(index, y_series):
    subjects = np.array(sorted(set(
        i.split("_", 1)[0] for i in index
        if i.split("_", 1)[0] not in HELD_OUT
    )))
    subj_labels = []
    for s in subjects:
        vals = y_series[[i for i in index if i.startswith(s + "_")]].values
        subj_labels.append(int(np.round(vals.mean())) if len(vals) else 1)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return [(set(subjects[tr]), set(subjects[te]))
            for tr, te in skf.split(subjects, subj_labels)]


ML_MODELS = [
    LogisticRegression(max_iter=1000, random_state=SEED),
    RandomForestClassifier(n_estimators=100, random_state=SEED),
    GradientBoostingClassifier(n_estimators=100, random_state=SEED),
    AdaBoostClassifier(n_estimators=100, random_state=SEED),
    SVC(random_state=SEED),
    KNeighborsClassifier(),
    DecisionTreeClassifier(random_state=SEED),
    GaussianNB(),
    MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=SEED),
]


def eval_ml(X_tr, y_tr, X_te, y_te, input_name, fold):
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_tr)
    Xs = scaler.transform(X_te)
    rows = []
    for clf in ML_MODELS:
        m = clone(clf)
        m.fit(Xt, y_tr)
        pred = m.predict(Xs)
        rows.append(dict(
            input=input_name, model=m.__class__.__name__, fold=fold,
            acc=accuracy_score(y_te, pred),
            bal_acc=balanced_accuracy_score(y_te, pred),
            f1w=f1_score(y_te, pred, average="weighted", zero_division=0),
            f1m=f1_score(y_te, pred, average="macro",    zero_division=0),
        ))
    return rows


class CrossModalTransformer(nn.Module):
    def __init__(self, phys_dim, video_dim, audio_dim, hidden=64):
        super().__init__()
        self.phys  = nn.Linear(phys_dim,  hidden)
        self.video = nn.Linear(video_dim, hidden)
        self.audio = nn.Linear(audio_dim, hidden)
        self.attn  = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 3, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, p, v, a):
        tokens = torch.stack([self.phys(p), self.video(v), self.audio(a)], dim=1)
        out, _ = self.attn(tokens, tokens, tokens)
        return self.fc(out.reshape(out.size(0), -1))


class AttentionFusionNet(nn.Module):
    def __init__(self, phys_dim, video_dim, audio_dim, hidden=64):
        super().__init__()
        self.phys  = nn.Linear(phys_dim,  hidden)
        self.video = nn.Linear(video_dim, hidden)
        self.audio = nn.Linear(audio_dim, hidden)
        self.att = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.Tanh(),
            nn.Linear(hidden, 3), nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, p, v, a):
        ep, ev, ea = self.phys(p), self.video(v), self.audio(a)
        w = self.att(torch.cat([ep, ev, ea], dim=1))
        return self.fc(w[:,0:1]*ep + w[:,1:2]*ev + w[:,2:3]*ea)


def eval_dl(X_tr, y_tr, X_te, y_te, phys_dim, video_dim, fold):
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_tr)
    Xs = scaler.transform(X_te)

    audio_dim = Xt.shape[1] - phys_dim - video_dim

    def make_tensors(X, y):
        p = torch.tensor(X[:, :phys_dim],              dtype=torch.float32)
        v = torch.tensor(X[:, phys_dim:phys_dim+video_dim], dtype=torch.float32)
        a = torch.tensor(X[:, phys_dim+video_dim:],    dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        return p, v, a, yt

    p_tr, v_tr, a_tr, y_tr_t = make_tensors(Xt, y_tr)
    p_te, v_te, a_te, y_te_t = make_tensors(Xs, y_te)

    train_loader = DataLoader(
        TensorDataset(p_tr, v_tr, a_tr, y_tr_t),
        batch_size=32, shuffle=True
    )

    rows = []
    for ModelClass in [CrossModalTransformer, AttentionFusionNet]:
        model = ModelClass(phys_dim, video_dim, audio_dim).to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit  = nn.CrossEntropyLoss()

        model.train()
        for _ in range(50):
            for pb, vb, ab, yb in train_loader:
                pb, vb, ab, yb = pb.to(DEVICE), vb.to(DEVICE), ab.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                crit(model(pb, vb, ab), yb).backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(p_te.to(DEVICE), v_te.to(DEVICE), a_te.to(DEVICE))
            pred = logits.argmax(1).cpu().numpy()

        rows.append(dict(
            input="Multimodal (DL)", model=ModelClass.__name__, fold=fold,
            acc=accuracy_score(y_te, pred),
            bal_acc=balanced_accuracy_score(y_te, pred),
            f1w=f1_score(y_te, pred, average="weighted", zero_division=0),
            f1m=f1_score(y_te, pred, average="macro",    zero_division=0),
        ))
    return rows


def run_cv(X, y, folds, phys_dim, video_dim):
    index = X.index.tolist()
    phys_end  = phys_dim
    video_end = phys_dim + video_dim
    all_rows  = []

    for fold_i, (train_subj, test_subj) in enumerate(folds):
        tr_idx = [i for i in index if i.split("_",1)[0] in train_subj]
        te_idx = [i for i in index if i.split("_",1)[0] in test_subj]

        X_tr = X.loc[tr_idx].values.astype(float)
        X_te = X.loc[te_idx].values.astype(float)
        y_tr = y.loc[tr_idx].values
        y_te = y.loc[te_idx].values

        if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
            continue

        for name, (s, e) in [
            ("Physiological",  (0,         phys_end)),
            ("Video",          (phys_end,  video_end)),
            ("Audio",          (video_end, X_tr.shape[1])),
            ("Feature Fusion", (0,         X_tr.shape[1])),
        ]:
            all_rows.extend(eval_ml(X_tr[:,s:e], y_tr, X_te[:,s:e], y_te, name, fold_i+1))

        all_rows.extend(eval_dl(X_tr, y_tr, X_te, y_te, phys_dim, video_dim, fold_i+1))

        print(f"  Fold {fold_i+1} done  (train={len(tr_idx)}, test={len(te_idx)})")

    return pd.DataFrame(all_rows)


def main():
    print(f"Device: {DEVICE}")
    print("Loading features...")
    X, y, phys_dim, video_dim = load_features()
    print(f"  Samples: {len(X)}  |  Classes: {y.value_counts().to_dict()}")
    print(f"  phys_dim={phys_dim}, video_dim={video_dim}, "
          f"audio_dim={X.shape[1]-phys_dim-video_dim}")

    folds = get_subject_splits(X.index.tolist(), y)
    print(f"  {N_FOLDS}-fold subject-level CV\n")

    df = run_cv(X, y, folds, phys_dim, video_dim)

    summary = (df.groupby(["input", "model"])[["acc", "f1m", "bal_acc"]]
               .mean().round(4)
               .sort_values("f1m", ascending=False))

    print("\n" + "="*72)
    print("  5-FOLD CV RESULTS (sorted by macro F1)")
    print("="*72)
    print(f"  {'Input':<22} {'Model':<30} {'Acc':>7} {'F1m':>7} {'BalAcc':>8}")
    print("  " + "-"*68)
    for (inp, mdl), row in summary.iterrows():
        print(f"  {inp:<22} {mdl:<30} {row['acc']:>7.4f} {row['f1m']:>7.4f} {row['bal_acc']:>8.4f}")

    print("\n" + "-"*72)
    print("  [Our model — train_fusion.py, 5-fold, Audio+Face+Gesture deep features]")
    print(f"  {'Late Fusion':<22} {'audio+face+gesture (deep)':<30}  0.8086  0.7565  0.7430")
    print("="*72)

    out = {}
    for (inp, mdl), grp in df.groupby(["input", "model"]):
        out[f"{inp} | {mdl}"] = {
            "acc_mean":      round(grp["acc"].mean(), 4),
            "f1m_mean":      round(grp["f1m"].mean(), 4),
            "f1w_mean":      round(grp["f1w"].mean(), 4),
            "bal_acc_mean":  round(grp["bal_acc"].mean(), 4),
            "acc_std":       round(grp["acc"].std(), 4),
            "f1m_std":       round(grp["f1m"].std(), 4),
        }
    out_path = os.path.join(BASE_DIR, "baseline_cv_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
