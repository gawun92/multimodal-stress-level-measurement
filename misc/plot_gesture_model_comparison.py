"""
plot_gesture_model_comparison.py

Create comparison plots for two gesture benchmark reports:
    1. Accuracy-only comparison
    2. Grouped comparison for other headline metrics

Usage:
    python plot_gesture_model_comparison.py \
        --pretrained-report results/gesture/benchmark_gesture_branch_fold0_binary-stress_20260412_195158_test_20260412_195354.txt \
        --stressid-report results/gesture/benchmark_gesture_branch_fold0_binary-stress_20260412_192942_test_20260412_193204.txt
"""

import argparse
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_OUTPUT_DIR = os.path.join("results", "gesture")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for two gesture benchmark reports."
    )
    parser.add_argument(
        "--pretrained-report",
        required=True,
        help="Path to the higher-accuracy pretrained gesture benchmark report.",
    )
    parser.add_argument(
        "--stressid-report",
        required=True,
        help="Path to the StressId-only gesture benchmark report.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the comparison plots will be written.",
    )
    return parser.parse_args()


def extract_metric(report_text, label):
    pattern = rf"^{re.escape(label)}:\s+([0-9]*\.?[0-9]+)$"
    match = re.search(pattern, report_text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find metric '{label}' in benchmark report.")
    return float(match.group(1))


def parse_report(path):
    with open(path, "r", encoding="ascii") as f:
        text = f.read()

    return {
        "accuracy": extract_metric(text, "Accuracy"),
        "weighted_f1": extract_metric(text, "Weighted F1"),
        "macro_f1": extract_metric(text, "Macro F1"),
        "auc_roc": extract_metric(text, "AUC-ROC"),
    }


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def save_accuracy_plot(path, pretrained_metrics, stressid_metrics):
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(7, 5))

    # labels = ["Pretrained", "StressId-only"]
    # values = [pretrained_metrics["accuracy"], stressid_metrics["accuracy"]]
    
    labels = ["StressId-only", "Pretrained"]
    values = [stressid_metrics["accuracy"], pretrained_metrics["accuracy"]]
    
    bars = ax.bar(
        labels,
        values,
        color=["#1565C0", "#EF6C00"],
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Gesture Model Accuracy Comparison", fontweight="bold")
    annotate_bars(ax, bars)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_other_metrics_plot(path, pretrained_metrics, stressid_metrics):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    metric_names = ["Weighted F1", "Macro F1", "AUC-ROC"]
    pretrained_values = [
        pretrained_metrics["weighted_f1"],
        pretrained_metrics["macro_f1"],
        pretrained_metrics["auc_roc"],
    ]
    stressid_values = [
        stressid_metrics["weighted_f1"],
        stressid_metrics["macro_f1"],
        stressid_metrics["auc_roc"],
    ]

    x = range(len(metric_names))
    width = 0.36
    
    stressid_bars = ax.bar(
        [idx - width / 2 for idx in x],
        stressid_values,
        width=width,
        label="StressId-only",
        color="#EF6C00",
        edgecolor="black",
        linewidth=0.6,
    )
    pretrained_bars = ax.bar(
        [idx + width / 2 for idx in x],
        pretrained_values,
        width=width,
        label="Pretrained",
        color="#1565C0",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Gesture Model Metric Comparison", fontweight="bold")
    ax.legend()

    annotate_bars(ax, stressid_bars)
    annotate_bars(ax, pretrained_bars)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    pretrained_metrics = parse_report(args.pretrained_report)
    stressid_metrics = parse_report(args.stressid_report)

    save_accuracy_plot(
        os.path.join(
            args.output_dir,
            f"gesture_model_accuracy_comparison_{run_ts}.png",
        ),
        pretrained_metrics,
        stressid_metrics,
    )
    save_other_metrics_plot(
        os.path.join(
            args.output_dir,
            f"gesture_model_other_metrics_comparison_{run_ts}.png",
        ),
        pretrained_metrics,
        stressid_metrics,
    )


if __name__ == "__main__":
    main()
