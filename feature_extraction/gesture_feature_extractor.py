"""
gesture_feature_extractor.py

Extract gesture landmarks from raw StressID videos using MediaPipe Pose
Landmarker. Videos are sampled at a fixed FPS and saved as fixed-length landmark
tensors with per-frame detection masks and clip-level metadata.

Selected landmarks:
    - head anchors: nose, left ear, right ear
    - upper body: left/right shoulder, left/right elbow, left/right wrist
    - torso anchors: left/right hip

Outputs per video:
    - `*_gesture.npy`:      landmark tensor of shape (MAX_FRAMES, 11, 3)
    - `*_gesture_mask.npy`: detection mask of shape (MAX_FRAMES,)
    - `*_gesture_meta.npz`: sampled frame count, detected frame count, coverage,
                              source fps, and source path

Normalization:
    - center on the midpoint between left and right shoulders
    - scale by max(shoulder width, torso length)
    - rotate around the z-axis so shoulders are horizontally aligned in image space

Usage:
    python feature_extraction/gesture_feature_extractor.py --split train
    python feature_extraction/gesture_feature_extractor.py --video-list-csv results/gesture/usable_gesture_videos_0p1.csv --labeled-only --skip-baseline
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


TARGET_FPS = 5
MAX_FRAMES = 300
VISIBILITY_THRESHOLD = 0.5

POSE_LANDMARK_INDICES = [
    0,   # nose
    7,   # left ear
    8,   # right ear
    11, 12,  # shoulders
    13, 14,  # elbows
    15, 16,  # wrists
    23, 24,  # hips
]
N_LANDMARKS = len(POSE_LANDMARK_INDICES)

NOSE_IDX = 0
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARC_DATA_DIR = "/project2/msoleyma_1026/group_14/data/stressid"
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "data", "stressid")
DATA_DIR = CARC_DATA_DIR if os.path.exists(CARC_DATA_DIR) else LOCAL_DATA_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "feature_extraction", "results", "gesture")
MODEL_PATH = os.path.join(BASE_DIR, "feature_extraction", "tasks", "pose_landmarker.task")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")


def _safe_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


def parse_video_identity(video_path: Path):
    stem = video_path.stem
    subject_id = stem.split("_")[0]
    task = "_".join(stem.split("_")[1:])
    return subject_id, task


def load_labeled_pairs(labels_csv: str):
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    labels_df = pd.read_csv(labels_csv)
    return set(labels_df["subject/task"].astype(str))


def resolve_video_paths(video_dir: str, video_list_csv: str = None):
    if video_list_csv is None:
        return sorted(Path(video_dir).glob("**/*.mp4"))

    df = pd.read_csv(video_list_csv)
    candidate_cols = ["video_path", "path", "video_name", "filename"]
    source_col = next((col for col in candidate_cols if col in df.columns), None)
    if source_col is None:
        raise ValueError(
            f"Video list CSV must contain one of: {candidate_cols}"
        )

    paths = []
    base_dir = Path(video_dir)
    has_subject_id = "subject_id" in df.columns

    for row in df.to_dict("records"):
        value = str(row[source_col])
        candidate = Path(value)

        if candidate.is_absolute():
            resolved = candidate
        elif has_subject_id and source_col in {"video_name", "filename"}:
            resolved = base_dir / str(row["subject_id"]) / candidate.name
        else:
            resolved = base_dir / candidate

        if not resolved.exists():
            matches = list(base_dir.glob(f"**/{candidate.name}"))
            if len(matches) == 1:
                resolved = matches[0]

        paths.append(resolved)
    return paths


def _frame_is_valid(visibilities: np.ndarray) -> bool:
    required = [
        POSE_LANDMARK_INDICES.index(LEFT_SHOULDER_IDX),
        POSE_LANDMARK_INDICES.index(RIGHT_SHOULDER_IDX),
    ]
    if np.any(visibilities[required] < VISIBILITY_THRESHOLD):
        return False

    anchor_candidates = [
        POSE_LANDMARK_INDICES.index(NOSE_IDX),
        POSE_LANDMARK_INDICES.index(LEFT_HIP_IDX),
        POSE_LANDMARK_INDICES.index(RIGHT_HIP_IDX),
    ]
    return np.any(visibilities[anchor_candidates] >= VISIBILITY_THRESHOLD)


def canonicalize_frame(frame: np.ndarray) -> np.ndarray:
    centered = frame.astype(np.float32, copy=True)

    left_shoulder = centered[POSE_LANDMARK_INDICES.index(LEFT_SHOULDER_IDX)]
    right_shoulder = centered[POSE_LANDMARK_INDICES.index(RIGHT_SHOULDER_IDX)]
    left_hip = centered[POSE_LANDMARK_INDICES.index(LEFT_HIP_IDX)]
    right_hip = centered[POSE_LANDMARK_INDICES.index(RIGHT_HIP_IDX)]

    shoulder_center = 0.5 * (left_shoulder + right_shoulder)
    hip_center = 0.5 * (left_hip + right_hip)
    centered -= shoulder_center

    shoulder_vec = right_shoulder - left_shoulder
    shoulder_width = np.linalg.norm(shoulder_vec[:2])
    torso_length = np.linalg.norm(shoulder_center[:2] - hip_center[:2])
    scale = max(shoulder_width, torso_length, 1e-8)
    centered /= scale

    x_axis = _safe_normalize(np.array([shoulder_vec[0], shoulder_vec[1]], dtype=np.float32))
    if np.linalg.norm(x_axis) < 1e-8:
        return centered

    angle = np.arctan2(x_axis[1], x_axis[0])
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    rotation = np.array(
        [[cos_a, -sin_a], [sin_a, cos_a]],
        dtype=np.float32,
    )
    centered[:, :2] = centered[:, :2] @ rotation.T
    return centered


def normalize_landmarks(landmarks: np.ndarray, detect_mask: np.ndarray) -> np.ndarray:
    normalized = landmarks.copy()
    for t in range(landmarks.shape[0]):
        if detect_mask[t] == 0:
            continue
        frame = landmarks[t]
        if frame.max() == 0 and frame.min() == 0:
            continue
        normalized[t] = canonicalize_frame(frame)
    return normalized


def extract_gesture_landmarks(mp4_path: str, model_path: str):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {mp4_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 15.0
    interval = max(1, int(round(original_fps / TARGET_FPS)))

    result = np.zeros((MAX_FRAMES, N_LANDMARKS, 3), dtype=np.float32)
    detect_mask = np.zeros((MAX_FRAMES,), dtype=np.uint8)
    frame_idx = 0
    saved_idx = 0

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
        while saved_idx < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(round(frame_idx * 1000.0 / original_fps))
                detection = landmarker.detect_for_video(mp_image, timestamp_ms)

                if detection.pose_landmarks:
                    all_landmarks = detection.pose_landmarks[0]
                    visibilities = np.zeros((N_LANDMARKS,), dtype=np.float32)
                    for out_idx, pose_idx in enumerate(POSE_LANDMARK_INDICES):
                        landmark = all_landmarks[pose_idx]
                        result[saved_idx, out_idx] = [landmark.x, landmark.y, landmark.z]
                        visibilities[out_idx] = landmark.visibility
                    if _frame_is_valid(visibilities):
                        detect_mask[saved_idx] = 1

                if detect_mask[saved_idx] == 0:
                    result[saved_idx] = 0.0

                saved_idx += 1

            frame_idx += 1

    cap.release()

    return result, detect_mask, saved_idx, original_fps


def process_all(split: str = "train", output_subdir: str = None,
                video_list_csv: str = None, labeled_only: bool = False,
                skip_baseline: bool = False, video_dir: str = None,
                labels_csv: str = LABELS_CSV,
                model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pose landmarker task file not found: {model_path}"
        )

    if video_dir is None:
        flat_video_dir = os.path.join(DATA_DIR, "Videos")
        has_flat_videos = os.path.exists(flat_video_dir)
        video_dir = flat_video_dir if has_flat_videos else os.path.join(DATA_DIR, split, "videos")
    else:
        has_flat_videos = False

    if has_flat_videos and video_list_csv is None:
        print(f"[gesture_feature_extractor] Using flat video directory: {video_dir}")
        print(
            f"[gesture_feature_extractor] Note: --split={split} does not filter source videos in this layout."
        )
    elif video_list_csv is not None:
        print(f"[gesture_feature_extractor] Using explicit video root: {video_dir}")

    if output_subdir is None:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        split_tag = split or "all"
        output_subdir = f"{split_tag}_{run_ts}"
    output_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    mp4_files = resolve_video_paths(video_dir, video_list_csv=video_list_csv)
    if not mp4_files:
        print(f"[gesture_feature_extractor] No .mp4 files found in {video_dir}")
        return

    labeled_pairs = load_labeled_pairs(labels_csv) if labeled_only else None
    success = 0
    skip = 0
    fail = 0

    for mp4_path in tqdm(mp4_files, desc=f"  [{output_subdir}] Gesture"):
        if not mp4_path.exists():
            fail += 1
            print(f"  [WARN] Missing video path: {mp4_path}")
            continue

        subject_id, task = parse_video_identity(mp4_path)
        key = f"{subject_id}_{task}"

        if skip_baseline and task == "Baseline":
            skip += 1
            continue
        if labeled_pairs is not None and key not in labeled_pairs:
            skip += 1
            continue

        subject_out_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_out_dir, exist_ok=True)

        landmarks_path = os.path.join(subject_out_dir, f"{task}_gesture.npy")
        mask_path = os.path.join(subject_out_dir, f"{task}_gesture_mask.npy")
        meta_path = os.path.join(subject_out_dir, f"{task}_gesture_meta.npz")

        if os.path.exists(landmarks_path) and os.path.exists(mask_path) and os.path.exists(meta_path):
            skip += 1
            continue

        try:
            landmarks, detect_mask, sampled_frames, source_fps = extract_gesture_landmarks(
                str(mp4_path),
                model_path=model_path,
            )
            landmarks = normalize_landmarks(landmarks, detect_mask)
            detected_frames = int(detect_mask.sum())
            coverage = (detected_frames / sampled_frames) if sampled_frames > 0 else 0.0

            np.save(landmarks_path, landmarks)
            np.save(mask_path, detect_mask)
            np.savez(
                meta_path,
                sampled_frames=np.int32(sampled_frames),
                detected_frames=np.int32(detected_frames),
                coverage=np.float32(coverage),
                source_fps=np.float32(source_fps),
                source_path=str(mp4_path),
            )
            success += 1
        except Exception as exc:
            fail += 1
            print(f"  [WARN] Failed: {mp4_path.name} -- {exc}")

    print(f"[gesture_feature_extractor] Done -- success: {success} | skipped: {skip} | failed: {fail}")
    print(f"[gesture_feature_extractor] Output saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract gesture landmarks from StressID videos")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--output-subdir", type=str, default=None)
    parser.add_argument("--video-list-csv", type=str, default=None)
    parser.add_argument("--labeled-only", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Optional explicit video root; defaults to StressID video dir")
    parser.add_argument("--labels-csv", type=str, default=LABELS_CSV)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    process_all(
        split=args.split,
        output_subdir=args.output_subdir,
        video_list_csv=args.video_list_csv,
        labeled_only=args.labeled_only,
        skip_baseline=args.skip_baseline,
        video_dir=args.video_dir,
        labels_csv=args.labels_csv,
        model_path=args.model_path,
    )
