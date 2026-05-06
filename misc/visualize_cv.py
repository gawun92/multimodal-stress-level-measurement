"""
visualize_cv.py

Generate presentation-ready visuals from the completed 5-fold CV run.
Reads from cv_results.json (produced by run_all_folds.py).

Usage:
    python visualize_cv.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, balanced_accuracy_score

import config

OUT_DIR = os.path.join(config.BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_JSON = os.path.join(config.BASE_DIR, "cv_results.json")
sns.set_theme(style="whitegrid", font_scale=1.2)

FOLD_COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]


def load_results():
    with open(RESULTS_JSON) as f:
        return json.load(f)


# ─────────────────────────────────────────
# 1. Per-Fold Metrics Bar Chart
# ─────────────────────────────────────────
def plot_fold_metrics(data):
    results = data["fold_results"]
    folds = [r["fold"] for r in results]
    accs   = [r["accuracy"]   for r in results]
    f1w    = [r["f1_weighted"] for r in results]
    f1m    = [r["f1_macro"]   for r in results]
    aucs   = [r["auc_roc"]    for r in results]

    x = np.arange(len(folds))
    w = 0.2

    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - 1.5*w, accs, w, label="Accuracy",      color="#4CAF50", edgecolor="black", lw=0.5)
    b2 = ax.bar(x - 0.5*w, f1w,  w, label="Weighted F1",   color="#2196F3", edgecolor="black", lw=0.5)
    b3 = ax.bar(x + 0.5*w, f1m,  w, label="Macro F1",      color="#FF9800", edgecolor="black", lw=0.5)
    b4 = ax.bar(x + 1.5*w, aucs, w, label="AUC-ROC",       color="#9C27B0", edgecolor="black", lw=0.5)

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Mean lines
    for vals, color, ls in [
        (accs, "#4CAF50", "--"), (f1w, "#2196F3", "--"),
        (f1m, "#FF9800", "--"), (aucs, "#9C27B0", "--")
    ]:
        ax.axhline(np.mean(vals), color=color, linestyle=ls, alpha=0.4, lw=1.2)

    # Baseline reference
    ax.axhline(0.68, color="red", linestyle=":", lw=2, label="SVM Baseline F1 (0.68)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Fold Test Metrics — 5-Fold Subject-Level CV", fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)

    # Summary text box
    summary = (
        f"Mean Acc:   {np.mean(accs):.3f} ± {np.std(accs):.3f}\n"
        f"Mean F1w:   {np.mean(f1w):.3f} ± {np.std(f1w):.3f}\n"
        f"Mean F1m:   {np.mean(f1m):.3f} ± {np.std(f1m):.3f}\n"
        f"Mean AUC:   {np.mean(aucs):.3f} ± {np.std(aucs):.3f}"
    )
    ax.text(0.01, 0.04, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cv_fold_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 2. Training Curves — All 5 Folds
# ─────────────────────────────────────────
def plot_training_curves(data):
    histories = data["histories"]

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))

    for i, (fold_str, h) in enumerate(sorted(histories.items())):
        fold = int(fold_str)
        epochs = range(1, len(h["train_loss"]) + 1)
        color = FOLD_COLORS[fold]

        # Loss
        ax_loss = axes[0, fold]
        ax_loss.plot(epochs, h["train_loss"], "b-", lw=2, label="Train")
        ax_loss.plot(epochs, h["val_loss"],   "r-", lw=2, label="Val")
        ax_loss.set_title(f"Fold {fold} — Loss", fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=9)

        # Accuracy
        ax_acc = axes[1, fold]
        ax_acc.plot(epochs, h["val_acc"], color=color, lw=2, marker="o", markersize=2)
        ax_acc.axhline(0.5, color="gray", linestyle="--", alpha=0.5, lw=1)
        ax_acc.set_ylim(0, 1)
        ax_acc.set_title(f"Fold {fold} — Val Accuracy", fontweight="bold")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")

        # Annotate final val acc
        final_acc = h["val_acc"][-1]
        ax_acc.annotate(f"{final_acc:.2f}", xy=(len(epochs), final_acc),
                        xytext=(-15, 8), textcoords="offset points",
                        fontsize=9, fontweight="bold", color=color)

    fig.suptitle("Training Curves — All 5 Folds (with SpecAugment)", fontweight="bold", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cv_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 3. Confusion Matrices — All 5 Folds
# ─────────────────────────────────────────
def plot_confusion_matrices(data):
    results = data["fold_results"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for r in results:
        fold = r["fold"]
        cm = np.array(r["confusion_matrix"])
        acc = r["accuracy"]
        auc_v = r["auc_roc"]

        ax = axes[fold]
        # Normalize for color (row-normalize = recall-normalized)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        sns.heatmap(cm_norm, annot=cm, fmt="d", ax=ax, cmap="Blues",
                    vmin=0, vmax=1, linewidths=0.5,
                    xticklabels=["No Stress", "Stressed"],
                    yticklabels=["No Stress", "Stressed"],
                    cbar=(fold == 4))
        ax.set_title(f"Fold {fold}\nAcc={acc:.2f} | AUC={auc_v:.2f}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual" if fold == 0 else "")

    fig.suptitle("Confusion Matrices — 5-Fold CV (counts, color=recall-normalized)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cv_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 4. ROC Curves — All 5 Folds
# ─────────────────────────────────────────
def plot_roc_curves(data):
    results = data["fold_results"]

    fig, ax = plt.subplots(figsize=(8, 7))

    for r in results:
        fold = r["fold"]
        labels = np.array(r["labels"])
        probs  = np.array(r["probs"])[:, 1]  # prob of "stressed" class
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=FOLD_COLORS[fold], lw=2,
                label=f"Fold {fold} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC = 0.500)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All 5 Folds", fontweight="bold")
    ax.legend(loc="lower right")

    # Mean AUC annotation
    mean_auc = np.mean([r["auc_roc"] for r in results])
    std_auc  = np.std([r["auc_roc"] for r in results])
    ax.text(0.55, 0.12, f"Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}",
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cv_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 5. Summary vs Baseline Comparison
# ─────────────────────────────────────────
def plot_baseline_comparison(data):
    results = data["fold_results"]
    f1w_vals  = [r["f1_weighted"] for r in results]
    f1m_vals  = [r["f1_macro"]   for r in results]
    auc_vals  = [r["auc_roc"]    for r in results]
    acc_vals  = [r["accuracy"]   for r in results]

    # Baseline (StressID paper, binary stress, SVM + handcrafted features)
    baseline_f1w = 0.68
    baseline_acc = 0.68  # approximate from paper

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ["Accuracy", "Weighted F1", "Macro F1", "AUC-ROC"]
    our_means = [np.mean(acc_vals), np.mean(f1w_vals), np.mean(f1m_vals), np.mean(auc_vals)]
    our_stds  = [np.std(acc_vals),  np.std(f1w_vals),  np.std(f1m_vals),  np.std(auc_vals)]
    baseline  = [baseline_acc, baseline_f1w, None, None]

    x = np.arange(len(metrics))
    bars = ax.bar(x, our_means, yerr=our_stds, color="#2196F3", width=0.5,
                  edgecolor="black", lw=0.8, capsize=6,
                  error_kw={"elinewidth": 2, "ecolor": "black"},
                  label="Our Model (CNN+Transformer, 5-fold CV)")

    for bar, mean, std in zip(bars, our_means, our_stds):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.02,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Baseline markers where available
    for i, b in enumerate(baseline):
        if b is not None:
            ax.scatter(i, b, color="red", zorder=5, s=120, marker="D", label="SVM Baseline (paper)" if i == 0 else "")
            ax.text(i + 0.27, b + 0.01, f"{b:.2f}", color="red", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Our Model vs. StressID Baseline\n(Subject-Level 5-Fold CV)", fontweight="bold")
    ax.legend(loc="upper right")
    ax.text(2.5, 0.05, "Note: Baseline uses no subject-level split\n(may have speaker leakage)",
            fontsize=9, color="gray", ha="center")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cv_vs_baseline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 6. Class Recall Per Fold (Minority Class Tracking)
# ─────────────────────────────────────────
def plot_class_recall(data):
    results = data["fold_results"]

    no_stress_recall, stressed_recall = [], []
    for r in results:
        cm = np.array(r["confusion_matrix"])  # [[TN, FP], [FN, TP]]
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]
        ns_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        s_recall  = tp / (tp + fn) if (tp + fn) > 0 else 0
        no_stress_recall.append(ns_recall)
        stressed_recall.append(s_recall)

    folds = [r["fold"] for r in results]
    x = np.arange(len(folds))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, no_stress_recall, w, label="No Stress (minority)",
                color="#4CAF50", edgecolor="black", lw=0.5)
    b2 = ax.bar(x + w/2, stressed_recall,  w, label="Stressed (majority)",
                color="#F44336", edgecolor="black", lw=0.5)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.0%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall Per Fold\n(Key: Can we detect no-stress minority class?)", fontweight="bold")
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, lw=1.2)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cv_class_recall.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# 7. Professor Dashboard — single comprehensive figure
# ─────────────────────────────────────────
def plot_professor_dashboard(data):
    """
    One-page summary for the professor with 6 panels:
      Panel 1 (top, full width) : Per-fold grouped bars — 5 metrics
      Panel 2 (mid-left x2)    : Class-recall heatmap (minority vs majority per fold)
      Panel 3 (mid-right)      : Mean ± Std summary table vs baselines
      Panel 4 (bot-left)       : Held-out confusion matrix
      Panel 5 (bot-mid)        : MCC + Balanced Accuracy per fold
      Panel 6 (bot-right)      : Radar chart — mean vs best-fold performance profile
    """
    results  = data["fold_results"]
    held_out = data.get("held_out_result")

    # ── Compute MCC + BalAcc from stored preds/labels ────────────────
    for r in results:
        lbls = np.array(r["labels"])
        prds = np.array(r["preds"])
        r["mcc"]     = matthews_corrcoef(lbls, prds)
        r["bal_acc"] = balanced_accuracy_score(lbls, prds)

    if held_out:
        ho_lbls = np.array(held_out["labels"])
        ho_prds = np.array(held_out["preds"])
        held_out["mcc"]     = matthews_corrcoef(ho_lbls, ho_prds)
        held_out["bal_acc"] = balanced_accuracy_score(ho_lbls, ho_prds)

    # ── Aggregate ─────────────────────────────────────────────────────
    folds    = [r["fold"]        for r in results]
    accs     = [r["accuracy"]    for r in results]
    f1w      = [r["f1_weighted"] for r in results]
    f1m      = [r["f1_macro"]    for r in results]
    aucs     = [r["auc_roc"]     for r in results]
    mccs     = [r["mcc"]         for r in results]
    bal_accs = [r["bal_acc"]     for r in results]

    # Best fold by Macro F1 (same criterion as run_all_folds.py)
    best_fold     = int(np.argmax(f1m))
    collapsed_fold = int(np.argmin(aucs))  # worst AUC = most collapsed

    # Per-class recall from confusion matrices
    ns_recall, s_recall = [], []
    for r in results:
        cm = np.array(r["confusion_matrix"])
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]
        ns_recall.append(tn / (tn + fp + 1e-8))
        s_recall.append(tp  / (tp + fn + 1e-8))

    # ── Color palette ─────────────────────────────────────────────────
    C = {
        "acc":     "#4CAF50",
        "f1w":     "#2196F3",
        "f1m":     "#FF9800",
        "auc":     "#9C27B0",
        "mcc":     "#00BCD4",
        "bal_acc": "#E91E63",
    }

    # ── Figure layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#f4f6f9")
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.38,
                          top=0.92, bottom=0.04, left=0.06, right=0.97)

    fig.suptitle(
        "Audio Branch — Professor Dashboard  ·  2D-CNN + Temporal Transformer  ·  Binary Stress\n"
        f"StressID | 50 CV subjects (4 held-out: {data.get('held_out_subjects', [])}) | "
        f"StratifiedKFold {len(folds)}-fold | SpecAugment ON | LR=5e-5 | WD=1e-3",
        fontsize=12.5, fontweight="bold", y=0.975, color="#1a1a2e",
    )

    # ═══════════════════════════════════════════════════════════════════
    # Panel 1 (top, full width): Per-fold grouped bars — 5 main metrics
    # ═══════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, :])

    metrics_list = [
        ("Accuracy",    accs, C["acc"]),
        ("Weighted F1", f1w,  C["f1w"]),
        ("Macro F1",    f1m,  C["f1m"]),
        ("AUC-ROC",     aucs, C["auc"]),
        ("Balanced Acc", bal_accs, C["bal_acc"]),
    ]
    n_m = len(metrics_list)
    x   = np.arange(len(folds))
    w   = 0.15
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * w

    for i, (name, vals, color) in enumerate(metrics_list):
        bars = ax1.bar(x + offsets[i], vals, w * 0.88, label=name, color=color,
                       edgecolor="white", lw=0.5, alpha=0.88)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.006,
                     f"{h:.2f}", ha="center", va="bottom",
                     fontsize=7, fontweight="bold", color="#222222")

    # Dashed mean lines
    for name, vals, color in metrics_list:
        ax1.axhline(np.mean(vals), color=color, linestyle="--", alpha=0.35, lw=1.2)

    # Highlight collapsed fold (red tint + label)
    ax1.axvspan(collapsed_fold - 0.48, collapsed_fold + 0.48,
                color="red", alpha=0.07, zorder=0)
    ax1.text(collapsed_fold, 1.03, "[!] Collapsed", ha="center", va="bottom",
             fontsize=9, color="#c62828", fontweight="bold",
             transform=ax1.get_xaxis_transform())

    # Highlight best fold (green tint + label)
    if best_fold != collapsed_fold:
        ax1.axvspan(best_fold - 0.48, best_fold + 0.48,
                    color="green", alpha=0.07, zorder=0)
        ax1.text(best_fold, 1.03, "[*] Best", ha="center", va="bottom",
                 fontsize=9, color="#2e7d32", fontweight="bold",
                 transform=ax1.get_xaxis_transform())

    # Reference lines
    ax1.axhline(0.68, color="red",  linestyle=":", lw=2,
                label="SVM Baseline F1w (0.68)", zorder=5)
    ax1.axhline(0.50, color="gray", linestyle="-", lw=0.8, alpha=0.45,
                label="Random (0.50)", zorder=4)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Fold {f}" for f in folds], fontsize=12)
    ax1.set_ylim(0, 1.09)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Per-Fold Test Metrics  (dashed = cross-fold mean; [!] = collapsed fold; [*] = best by Macro F1)",
                  fontweight="bold", fontsize=11, pad=6)
    ax1.legend(loc="upper right", fontsize=9, ncol=4, framealpha=0.88)
    ax1.set_facecolor("#fdfdfd")

    # ═══════════════════════════════════════════════════════════════════
    # Panel 2 (mid, spans 2 cols): Class-Recall Heatmap
    # ═══════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1, :2])

    recall_matrix = np.array([ns_recall, s_recall])  # shape (2, n_folds)
    im = ax2.imshow(recall_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(2):
        for j in range(len(folds)):
            val  = recall_matrix[i, j]
            col  = "white" if (val < 0.28 or val > 0.78) else "#111111"
            ax2.text(j, i, f"{val:.0%}", ha="center", va="center",
                     fontsize=14, fontweight="bold", color=col)

    # Red border around collapsed fold column
    ax2.add_patch(plt.Rectangle(
        (collapsed_fold - 0.5, -0.5), 1, 2,
        fill=False, edgecolor="#c62828", lw=2.5, zorder=5
    ))

    ax2.set_xticks(range(len(folds)))
    ax2.set_xticklabels([f"Fold {f}" for f in folds], fontsize=12)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(
        ["No-Stress  (minority ~29%)", "Stressed  (majority ~71%)"], fontsize=11
    )
    ax2.set_title(
        "Per-Class Recall Per Fold  —  Key Question: Can we detect the minority class?  "
        "(red border = collapsed fold)",
        fontweight="bold", fontsize=11, pad=7,
    )
    plt.colorbar(im, ax=ax2, fraction=0.025, pad=0.02, label="Recall")

    # ═══════════════════════════════════════════════════════════════════
    # Panel 3 (mid-right): Summary table
    # ═══════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis("off")
    ax3.set_facecolor("#f9f9f9")

    def status(mean, threshold, higher_is_better=True):
        ok = mean >= threshold if higher_is_better else mean <= threshold
        return ("✔" if ok else "✗"), ("#2e7d32" if ok else "#c62828")

    rows_data = [
        ("Metric",       "Mean ± Std",     "Ref",   ""),
        ("─" * 11,       "─" * 13,         "─" * 5, ""),
        ("Accuracy",     f"{np.mean(accs):.3f}±{np.std(accs):.3f}",     "0.68",  *status(np.mean(accs),     0.68)),
        ("Weighted F1",  f"{np.mean(f1w):.3f}±{np.std(f1w):.3f}",       "0.68",  *status(np.mean(f1w),      0.68)),
        ("Macro F1",     f"{np.mean(f1m):.3f}±{np.std(f1m):.3f}",       ">0.50", *status(np.mean(f1m),      0.50)),
        ("AUC-ROC",      f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}",     ">0.50", *status(np.mean(aucs),     0.50)),
        ("Balanced Acc", f"{np.mean(bal_accs):.3f}±{np.std(bal_accs):.3f}", ">0.50", *status(np.mean(bal_accs), 0.50)),
        ("MCC",          f"{np.mean(mccs):.3f}±{np.std(mccs):.3f}",     ">0",    *status(np.mean(mccs),     0.0)),
        ("─" * 11,       "─" * 13,         "─" * 5, ""),
        (f"Best fold",   f"#{best_fold}",  f"F1m={f1m[best_fold]:.3f}", ""),
    ]
    if held_out:
        rows_data.append((
            "Held-out",
            f"Acc={held_out['accuracy']:.3f}",
            f"AUC={held_out['auc_roc']:.3f}",
            "",
        ))

    ax3.text(0.5, 1.00, "Mean Performance Summary", ha="center", va="top",
             fontsize=11, fontweight="bold", transform=ax3.transAxes, color="#1a1a2e")
    ax3.text(0.5, 0.95, "(vs SVM baseline / random)", ha="center", va="top",
             fontsize=9, color="#555555", transform=ax3.transAxes)

    col_x   = [0.01, 0.42, 0.72, 0.90]
    y_start = 0.88
    row_h   = 0.082

    for i, row in enumerate(rows_data):
        y = y_start - i * row_h
        is_header  = (i == 0)
        is_divider = (i == 1 or i == len(rows_data) - 2 - (1 if held_out else 0))
        for j, (txt, xp) in enumerate(zip(row, col_x)):
            if j == 3 and txt in ("✔", "✗"):          # status column
                sc = row[3]                            # colour already a string
                # colour was packed as 4th element
                fc = "#2e7d32" if txt == "✔" else ("#c62828" if txt == "✗" else "#333333")
                ax3.text(xp, y, txt, ha="left", va="top",
                         fontsize=10, fontweight="bold", color=fc,
                         transform=ax3.transAxes, fontfamily="monospace")
            else:
                fc = ("#1a1a2e" if is_header
                      else "#888888" if is_divider
                      else "#333333")
                fw = "bold" if is_header else "normal"
                ax3.text(xp, y, str(txt), ha="left", va="top",
                         fontsize=8.5, fontweight=fw, color=fc,
                         transform=ax3.transAxes, fontfamily="monospace")

    ax3.set_title("Mean Performance Summary", fontweight="bold", fontsize=11, pad=7)

    # ═══════════════════════════════════════════════════════════════════
    # Panel 4 (bot-left): Held-out Confusion Matrix
    # ═══════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[2, 0])

    if held_out:
        ho_cm      = np.array(held_out["confusion_matrix"])
        ho_cm_norm = ho_cm.astype(float) / (ho_cm.sum(axis=1, keepdims=True) + 1e-8)
        sns.heatmap(ho_cm_norm, annot=ho_cm, fmt="d", ax=ax4, cmap="Blues",
                    vmin=0, vmax=1, linewidths=1,
                    xticklabels=["No Stress", "Stressed"],
                    yticklabels=["No Stress", "Stressed"],
                    cbar=True, annot_kws={"size": 15, "weight": "bold"})
        ax4.set_title(
            f"Held-Out: {held_out['subjects']}\n"
            f"Acc={held_out['accuracy']:.3f} | F1m={held_out['f1_macro']:.3f} | "
            f"AUC={held_out['auc_roc']:.3f} | MCC={held_out['mcc']:.3f}",
            fontweight="bold", fontsize=9.5,
        )
        ax4.set_xlabel("Predicted")
        ax4.set_ylabel("Actual")
    else:
        ax4.text(0.5, 0.5, "No held-out data available",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=10)
        ax4.set_title("Held-Out (Not Available)", fontweight="bold")

    # ═══════════════════════════════════════════════════════════════════
    # Panel 5 (bot-mid): MCC + Balanced Accuracy per fold
    # ═══════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#fdfdfd")

    x5 = np.arange(len(folds))
    w5 = 0.35
    b1 = ax5.bar(x5 - w5 / 2, mccs,     w5, label="MCC",
                 color=C["mcc"],     edgecolor="black", lw=0.5, alpha=0.85)
    b2 = ax5.bar(x5 + w5 / 2, bal_accs, w5, label="Balanced Accuracy",
                 color=C["bal_acc"], edgecolor="black", lw=0.5, alpha=0.85)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2,
                 h + 0.02 if h >= 0 else h - 0.07,
                 f"{h:.2f}", ha="center",
                 va="bottom" if h >= 0 else "top",
                 fontsize=9, fontweight="bold")

    ax5.axhline(0.0, color=C["mcc"],     linestyle=":", lw=1.8, alpha=0.7,
                label="MCC=0 (random)")
    ax5.axhline(0.5, color=C["bal_acc"], linestyle="--", lw=1.8, alpha=0.6,
                label="BalAcc=0.5 (random)")

    ax5.set_xticks(x5)
    ax5.set_xticklabels([f"Fold {f}" for f in folds])
    ax5.set_ylim(-0.65, 1.05)
    ax5.set_ylabel("Score")
    ax5.set_title(
        "MCC & Balanced Accuracy Per Fold\n"
        "(Gold-standard for imbalanced binary — collapse → MCC≈0, BalAcc≈0.5)",
        fontweight="bold", fontsize=10,
    )
    ax5.legend(fontsize=8, loc="upper left")

    # ═══════════════════════════════════════════════════════════════════
    # Panel 6 (bot-right): Radar chart — mean vs best-fold profile
    # ═══════════════════════════════════════════════════════════════════
    ax6 = fig.add_subplot(gs[2, 2], polar=True)

    radar_labels = ["Accuracy", "Weighted\nF1", "Macro\nF1",
                    "AUC-ROC", "Balanced\nAcc", "MCC\n(→[0,1])"]
    n_ax = len(radar_labels)

    def _to_radar(a, f1w_v, f1m_v, auc_v, ba_v, mcc_v):
        """Map all metrics to [0, 1]; MCC is shifted from [-1,1]."""
        return [a, f1w_v, f1m_v, auc_v, ba_v, (mcc_v + 1) / 2]

    mean_vals = _to_radar(
        np.mean(accs), np.mean(f1w), np.mean(f1m),
        np.mean(aucs), np.mean(bal_accs), np.mean(mccs),
    )
    best_vals = _to_radar(
        accs[best_fold], f1w[best_fold], f1m[best_fold],
        aucs[best_fold], bal_accs[best_fold], mccs[best_fold],
    )

    angles = np.linspace(0, 2 * np.pi, n_ax, endpoint=False).tolist()
    angles += angles[:1]

    mean_plot = mean_vals + mean_vals[:1]
    best_plot = best_vals + best_vals[:1]
    ref_plot  = [0.5] * n_ax + [0.5]

    # Reference circle at 0.5
    ax6.plot(angles, ref_plot, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax6.fill(angles, ref_plot, alpha=0.05, color="gray")

    # Mean across folds
    ax6.plot(angles, mean_plot, "o-", lw=2.5, color="#2196F3",
             label=f"Mean (5-fold)", markersize=5)
    ax6.fill(angles, mean_plot, alpha=0.18, color="#2196F3")

    # Best fold overlay
    ax6.plot(angles, best_plot, "s-", lw=2.5, color="#4CAF50",
             label=f"Best fold {best_fold}", markersize=5)
    ax6.fill(angles, best_plot, alpha=0.14, color="#4CAF50")

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(radar_labels, size=9)
    ax6.set_ylim(0, 1)
    ax6.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax6.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], size=7)
    ax6.set_title(
        f"Performance Profile\nMean vs Best Fold {best_fold}  (MCC norm. to [0,1])",
        fontweight="bold", fontsize=10, pad=20,
    )
    ax6.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18), fontsize=8.5)

    # ── Save ──────────────────────────────────────────────────────────
    path = os.path.join(OUT_DIR, "professor_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────
def main():
    if not os.path.exists(RESULTS_JSON):
        print(f"ERROR: {RESULTS_JSON} not found. Run run_all_folds.py first.")
        return

    print(f"[visualize_cv] Loading results from: {RESULTS_JSON}")
    data = load_results()
    print(f"[visualize_cv] Saving figures to: {OUT_DIR}")

    print("\n--- Per-Fold Metrics ---")
    plot_fold_metrics(data)

    print("\n--- Training Curves ---")
    plot_training_curves(data)

    print("\n--- Confusion Matrices ---")
    plot_confusion_matrices(data)

    print("\n--- ROC Curves ---")
    plot_roc_curves(data)

    print("\n--- Baseline Comparison ---")
    plot_baseline_comparison(data)

    print("\n--- Class Recall ---")
    plot_class_recall(data)

    print("\n--- Professor Dashboard ---")
    plot_professor_dashboard(data)

    print(f"\n[visualize_cv] Done! All figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
