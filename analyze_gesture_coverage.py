"""
analyze_gesture_coverage.py

Quantify gesture detection coverage per task on raw StressID videos.
For each video, sample frames at a target FPS, run MediaPipe Pose Landmarker,
and record the fraction of sampled frames where the gesture landmarks
are usable.

Usable frame rule:
    - left shoulder and right shoulder must both be visible
    - at least one anchor landmark must be visible:
        nose, left hip, or right hip

Outputs:
    - CSV with per-video coverage
    - TXT summary with per-task aggregates
    - PNG bar chart of mean coverage by task

Usage:
    python analyze_gesture_coverage.py
    python analyze_gesture_coverage.py --max-videos 100
    python analyze_gesture_coverage.py --tasks Counting1 Math Speaking
    python analyze_gesture_coverage.py --video-dir data/processed/Videos --dataset-name processed_gesture
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

BASE_DIR = config.BASE_DIR
DEFAULT_VIDEO_DIR = os.path.join(config.DATA_DIR, "Videos")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "feature_extraction", "tasks", "pose_landmarker.task")
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "results", "gesture")
DEFAULT_FIGURES_DIR = os.path.join(BASE_DIR, "figures")
VISIBILITY_THRESHOLD = 0.5

POSE_NOSE_IDX = 0
POSE_LEFT_SHOULDER_IDX = 11
POSE_RIGHT_SHOULDER_IDX = 12
POSE_LEFT_HIP_IDX = 23
POSE_RIGHT_HIP_IDX = 24


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze gesture coverage per task")
    parser.add_argument("--video-dir", type=str, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--target-fps", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--max-videos", type=int, default=0,
                        help="Process at most N videos; 0 means all videos")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Optional subset of tasks to analyze")
    parser.add_argument("--dataset-name", type=str, default="gesture",
                        help="Name used in output filenames")
    parser.add_argument("--visibility-threshold", type=float, default=VISIBILITY_THRESHOLD)
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=str, default=DEFAULT_FIGURES_DIR)
    return parser.parse_args()


def parse_video_identity(path):
    stem = path.stem
    subject_id = stem.split("_")[0]
    task = "_".join(stem.split("_")[1:])
    return subject_id, task


def iter_videos(video_dir, tasks=None, max_videos=0):
    video_paths = sorted(Path(video_dir).glob("**/*.mp4"))
    filtered = []
    allowed = set(tasks) if tasks else None
    for path in video_paths:
        _, task = parse_video_identity(path)
        if allowed and task not in allowed:
            continue
        filtered.append(path)
        if max_videos and len(filtered) >= max_videos:
            break
    return filtered


def frame_is_usable(landmarks, threshold):
    left_shoulder = landmarks[POSE_LEFT_SHOULDER_IDX]
    right_shoulder = landmarks[POSE_RIGHT_SHOULDER_IDX]
    if left_shoulder.visibility < threshold or right_shoulder.visibility < threshold:
        return False

    return any(
        landmarks[idx].visibility >= threshold
        for idx in (POSE_NOSE_IDX, POSE_LEFT_HIP_IDX, POSE_RIGHT_HIP_IDX)
    )


def compute_video_coverage(mp4_path, model_path, target_fps, max_frames, visibility_threshold):
    import cv2

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {mp4_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 15.0
    interval = max(1, int(round(original_fps / target_fps)))

    frame_idx = 0
    sampled_frames = 0
    detected_frames = 0

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    with PoseLandmarker.create_from_options(options) as landmarker:
        while sampled_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(round(frame_idx * 1000.0 / original_fps))
                detection = landmarker.detect_for_video(mp_image, timestamp_ms)
                sampled_frames += 1
                if detection.pose_landmarks and frame_is_usable(
                    detection.pose_landmarks[0],
                    visibility_threshold,
                ):
                    detected_frames += 1

            frame_idx += 1

    cap.release()

    coverage = (detected_frames / sampled_frames) if sampled_frames > 0 else 0.0
    return {
        "sampled_frames": sampled_frames,
        "detected_frames": detected_frames,
        "coverage": coverage,
    }


def summarize(df):
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby("task")
        .agg(
            videos=("video_name", "count"),
            subjects=("subject_id", "nunique"),
            sampled_frames=("sampled_frames", "sum"),
            detected_frames=("detected_frames", "sum"),
            mean_coverage=("coverage", "mean"),
            median_coverage=("coverage", "median"),
            std_coverage=("coverage", "std"),
            min_coverage=("coverage", "min"),
            max_coverage=("coverage", "max"),
        )
        .reset_index()
        .sort_values("mean_coverage", ascending=False)
    )
    summary["std_coverage"] = summary["std_coverage"].fillna(0.0)
    summary["frame_weighted_coverage"] = (
        summary["detected_frames"] / summary["sampled_frames"].clip(lower=1)
    )
    return summary


def save_text_summary(path, summary_df, args, n_videos):
    lines = [
        "Gesture Coverage Summary",
        "=" * 72,
        f"Dataset name: {args.dataset_name}",
        f"Video directory: {args.video_dir}",
        f"Model path: {args.model_path}",
        f"Target FPS: {args.target_fps}",
        f"Max frames/video: {args.max_frames}",
        f"Visibility threshold: {args.visibility_threshold}",
        f"Videos analyzed: {n_videos}",
        "",
    ]

    if summary_df.empty:
        lines.append("No videos matched the requested filters.")
    else:
        lines.append("Per-task summary")
        lines.append("-" * 72)
        for row in summary_df.itertuples(index=False):
            lines.extend([
                (
                    f"{row.task}: videos={row.videos}, subjects={row.subjects}, "
                    f"mean={row.mean_coverage:.3f}, median={row.median_coverage:.3f}, "
                    f"frame_weighted={row.frame_weighted_coverage:.3f}, "
                    f"min={row.min_coverage:.3f}, max={row.max_coverage:.3f}"
                )
            ])

    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))


def save_plot(path, summary_df):
    if summary_df.empty:
        return

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        summary_df["task"],
        summary_df["mean_coverage"],
        yerr=summary_df["std_coverage"],
        color="#00897B",
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean Gesture Coverage")
    ax.set_xlabel("Task")
    ax.set_title("Gesture Coverage by Task", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)

    for bar, value in zip(bars, summary_df["mean_coverage"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Pose landmarker task file not found: {args.model_path}"
        )

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    video_paths = iter_videos(args.video_dir, tasks=args.tasks, max_videos=args.max_videos)
    if not video_paths:
        raise RuntimeError("No videos found for the requested filters.")

    rows = []
    for idx, video_path in enumerate(video_paths, start=1):
        subject_id, task = parse_video_identity(video_path)
        stats = compute_video_coverage(
            video_path,
            model_path=args.model_path,
            target_fps=args.target_fps,
            max_frames=args.max_frames,
            visibility_threshold=args.visibility_threshold,
        )
        rows.append({
            "subject_id": subject_id,
            "task": task,
            "video_name": video_path.name,
            "sampled_frames": stats["sampled_frames"],
            "detected_frames": stats["detected_frames"],
            "coverage": stats["coverage"],
        })
        print(
            f"[{idx:03d}/{len(video_paths):03d}] {video_path.name}: "
            f"coverage={stats['coverage']:.3f} "
            f"({stats['detected_frames']}/{stats['sampled_frames']})"
        )

    video_df = pd.DataFrame(rows)
    summary_df = summarize(video_df)

    stem = args.dataset_name
    csv_path = os.path.join(args.results_dir, f"{stem}_coverage_per_video.csv")
    txt_path = os.path.join(args.results_dir, f"{stem}_coverage_summary.txt")
    fig_path = os.path.join(args.figures_dir, f"{stem}_coverage_by_task.png")

    video_df.to_csv(csv_path, index=False)
    save_text_summary(txt_path, summary_df, args, len(video_paths))
    save_plot(fig_path, summary_df)

    print(f"[analyze_gesture_coverage] Per-video CSV saved: {csv_path}")
    print(f"[analyze_gesture_coverage] Summary TXT saved: {txt_path}")
    print(f"[analyze_gesture_coverage] Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
