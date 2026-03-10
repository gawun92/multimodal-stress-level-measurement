import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

TARGET_FPS = 5
MAX_FRAMES = 300

# Stress-relevant landmark indices only
# Optimized for EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio)
# total 86 landmarks
KEY_LANDMARKS = [
    # Left eye  — upper lid, lower lid, corners
    33, 160, 158, 133, 153, 144, 163, 7, 246, 161, 159, 157, 173,
    # Right eye — upper lid, lower lid, corners
    362, 385, 387, 263, 373, 380, 390, 249, 466, 388, 386, 384, 398,
    # Left eyebrow
    46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
    # Right eyebrow
    276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
    # Mouth outer upper lip
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    # Mouth outer lower lip
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    # Mouth inner upper lip
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    # Mouth inner lower lip
    95, 88, 178, 87, 14, 317, 402, 318, 324,
]
N_KEY_LANDMARKS = len(KEY_LANDMARKS)  # 86

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARC_DATA_DIR = "/project2/msoleyma_1026/group_14/data/stressid"
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "data", "stressid")
DATA_DIR = CARC_DATA_DIR if os.path.exists(CARC_DATA_DIR) else LOCAL_DATA_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "feature_extraction", "results", "face")
MODEL_PATH = os.path.join(BASE_DIR, "feature_extraction", "tasks", "face_landmarker.task")

def extract_face_landmarks(mp4_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {mp4_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 15.0
    interval = max(1, int(round(original_fps / TARGET_FPS)))

    result = np.zeros((MAX_FRAMES, N_KEY_LANDMARKS, 3), dtype=np.float32)
    frame_idx = 0
    saved_idx = 0

    # New mediapipe API
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        while saved_idx < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detection = landmarker.detect(mp_image)

                if detection.face_landmarks:
                    lm = detection.face_landmarks[0]
                    for new_j, orig_j in enumerate(KEY_LANDMARKS):
                        if orig_j < len(lm):
                            result[saved_idx, new_j] = [lm[orig_j].x, lm[orig_j].y, lm[orig_j].z]

                saved_idx += 1
            frame_idx += 1

    cap.release()
    return result


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    normalized = landmarks.copy()
    for t in range(landmarks.shape[0]):
        frame = landmarks[t]
        if frame.max() == 0:
            continue
        mins = frame.min(axis=0, keepdims=True)
        maxs = frame.max(axis=0, keepdims=True)
        normalized[t] = (frame - mins) / ((maxs - mins) + 1e-8)
    return normalized

def process_all(split: str = "train"):
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("Download it with:")
        print(
            "  curl -o feature_extraction/tasks/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        return

    video_dir = os.path.join(DATA_DIR, "Videos") if os.path.exists(os.path.join(DATA_DIR, "Videos")) else os.path.join(
        DATA_DIR, split, "videos")
    output_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(output_dir, exist_ok=True)

    mp4_files = sorted(Path(video_dir).glob("**/*.mp4"))
    if not mp4_files:
        print(f"[face_feature_extractor] No .mp4 files found in {video_dir}")
        return

    print(f"[face_feature_extractor] Found {len(mp4_files)} .mp4 files in '{split}' split")
    print(f"[face_feature_extractor] Extracting {N_KEY_LANDMARKS} key landmarks (86)")

    success, skip, fail = 0, 0, 0

    for mp4_path in tqdm(mp4_files, desc=f"  [{split}] Face Landmarks"):
        stem = mp4_path.stem
        subject_id = stem.split("_")[0]
        task = "_".join(stem.split("_")[1:])

        subject_out_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_out_dir, exist_ok=True)
        out_path = os.path.join(subject_out_dir, f"{task}_face.npy")

        if os.path.exists(out_path):
            skip += 1
            continue

        try:
            landmarks = extract_face_landmarks(str(mp4_path))
            landmarks = normalize_landmarks(landmarks)
            np.save(out_path, landmarks)
            success += 1
        except Exception as e:
            print(f"  [WARN] Failed: {mp4_path.name} — {e}")
            fail += 1

    print(f"[face_feature_extractor] Done — success: {success} | skipped: {skip} | failed: {fail}")
    print(f"[face_feature_extractor] Output shape: ({MAX_FRAMES}, {N_KEY_LANDMARKS}, 3)")
    print(f"[face_feature_extractor] Output saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    args = parser.parse_args()

    process_all(split=args.split)