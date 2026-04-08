"""
face feature extraction => extracts per-frame features from face videos:
  1. media pipe face landmarker => 92 arousal-relevant 3D landmarks (eyes, eyebrows, lips)
                              aligned to canonical face space
  2. stress-relevant AUs => 10 AU intensities computed
                              geometrically from landmarks (FACS-based)

usage:
    python feature_extraction/face_feature_extractor.py
    python feature_extraction/face_feature_extractor.py --split train
    python feature_extraction/face_feature_extractor.py --data_dir /path/to/data
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

# config
TARGET_FPS = 5
MAX_FRAMES = 300
N_LANDMARKS = 478

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARC_DATA_DIR = "/project2/msoleyma_1026/group_14/data/stressid"
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "data", "stressid")
DATA_DIR = CARC_DATA_DIR if os.path.exists(CARC_DATA_DIR) else LOCAL_DATA_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "feature_extraction", "results", "face")
MODEL_PATH = os.path.join(BASE_DIR, "feature_extraction", "tasks", "face_landmarker.task")

# Arousal-relevant landmark indices
# (eyes, eyebrows, lips only)
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        185, 40, 39, 37, 0, 267, 269, 270, 409,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        191, 80, 81, 82, 13, 312, 311, 310, 415]

SELECTED_LANDMARKS = LEFT_EYE + RIGHT_EYE + LEFT_BROW + RIGHT_BROW + LIPS
N_SELECTED = len(SELECTED_LANDMARKS)  # 92

# lookup: MediaPipe index => position in SELECTED array
LM_IDX = {mp_idx: i for i, mp_idx in enumerate(SELECTED_LANDMARKS)}

# Stress-relevant Action Units (FACS-based)
# Computed geometrically from landmarks
STRESS_AU_LABELS = [
    "AU1_inner_brow_raise",  # inner brow raised => distress
    "AU4_brow_lowerer",  # brow descends => frown/stress
    "AU5_upper_lid_raiser",  # eye aperture => arousal/alertness
    "AU7_lid_tightener",  # lid tightens => stress/tension
    "AU15_lip_corner_depr",  # corners drop => negative affect
    "AU20_lip_stretcher",  # lips stretch sideways => fear/stress
    "AU23_lip_tightener",  # lip compression => stress
    "AU24_lip_pressor",  # lips pressed together => suppression
    "AU25_lips_part",  # mouth opens => arousal
    "AU43_eye_closure",  # eyes closing => fatigue/low arousal
]
N_STRESS_AUS = 10


def compute_stress_aus(frame: np.ndarray) -> np.ndarray:
    # Compute 10 stress-relevant AU intensities geometrically from landmarks.
    if frame.max() == 0:
        return np.zeros(N_STRESS_AUS, dtype=np.float32)

    # build per-landmark lookup
    lm = {mp_idx: frame[LM_IDX[mp_idx]] for mp_idx in SELECTED_LANDMARKS}

    def dist(a, b):
        return np.linalg.norm(lm[a] - lm[b]) + 1e-8

    # reference scale: inter-eye distance
    eye_dist = dist(33, 263)

    # AU1
    # inner brow landmarks move away from eye corner when raised
    au1 = (dist(46, 33) + dist(276, 263)) / (2 * eye_dist)

    # AU4
    # brow descends toward eye  => smaller brow-eye distance = higher AU4
    brow_eye_l = dist(66, 159)
    brow_eye_r = dist(296, 386)
    au4 = max(0.0, 1.0 - (brow_eye_l + brow_eye_r) / (2 * eye_dist))

    # AU5 / AU43
    # Eye Aspect Ratio: vertical / horizontal
    ear_l = (dist(160, 144) + dist(158, 153)) / (2 * dist(33, 133))
    ear_r = (dist(387, 373) + dist(385, 380)) / (2 * dist(362, 263))
    ear = (ear_l + ear_r) / 2

    au5 = ear  # high EAR = wide open eyes
    au43 = max(0.0, 1.0 - ear * 4)  # high when eyes nearly closed

    # AU7
    # upper lid lowers slightly (inverse of full eye opening)
    au7 = max(0.0, 1.0 - ear * 2)

    # AU15
    # lip corners drop below mouth midpoint
    mouth_mid_y = (lm[0][1] + lm[17][1]) / 2
    drop_l = max(0.0, lm[61][1] - mouth_mid_y)
    drop_r = max(0.0, lm[291][1] - mouth_mid_y)
    au15 = (drop_l + drop_r) / (2 * eye_dist)

    # AU20
    # horizontal stretch of lips (corners move apart)
    au20 = dist(61, 291) / eye_dist

    # AU25
    # vertical mouth opening (inner lips)
    mouth_open = dist(13, 14)
    au25 = mouth_open / eye_dist

    # AU23
    # wide & closed lips  => high tightening ratio
    au23 = au20 / (au25 + 1e-6)

    # AU24
    # lips pressed shut  => low mouth opening
    au24 = max(0.0, 1.0 - au25 * 5)

    aus = np.array([au1, au4, au5, au7, au15, au20, au23, au24, au25, au43],
                   dtype=np.float32)
    return np.clip(aus, 0.0, None)


# canonical-space alignment
# using MediaPipe's facial_transformation_matrices
def to_canonical_space(landmarks: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    # transform landmarks from image space to MediaPipe canonical face space.
    T_inv = np.linalg.inv(transform_matrix)
    ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
    lm_h = np.hstack([landmarks, ones])
    canonical = (T_inv @ lm_h.T).T[:, :3]
    return canonical.astype(np.float32)


# mediaPipe: landmark extraction
def extract_face_features(mp4_path: str) -> np.ndarray:
    # Extract face landmarks + stress AUs from a video.
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {mp4_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 15.0
    interval = max(1, int(round(original_fps / TARGET_FPS)))

    feature_dim = N_SELECTED * 3 + N_STRESS_AUS
    result = np.zeros((MAX_FRAMES, feature_dim), dtype=np.float32)
    saved_idx = 0
    frame_idx = 0

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_facial_transformation_matrixes=True,
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

                if detection.face_landmarks and detection.facial_transformation_matrixes:
                    lm = detection.face_landmarks[0]
                    T = np.array(detection.facial_transformation_matrixes[0].data).reshape(4, 4)

                    # extract all 478 landmarks
                    raw_all = np.array([[lm[j].x, lm[j].y, lm[j].z]
                                        for j in range(min(len(lm), N_LANDMARKS))],
                                       dtype=np.float32)

                    # apply canonical-space alignment to all landmarks
                    canonical_all = to_canonical_space(raw_all, T)

                    # filter to arousal-relevant regions only (eyes, eyebrows, lips)
                    selected = canonical_all[SELECTED_LANDMARKS]

                    # compute stress-relevant AUs geometrically
                    aus = compute_stress_aus(selected)

                    # concatenate: landmarks + AUs
                    result[saved_idx] = np.concatenate([selected.flatten(), aus])

                # if no face detected  => row stays 0 (zero-padding)
                saved_idx += 1

            frame_idx += 1

    cap.release()
    return result


# batch processing
def process_all(split: str = "train"):
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] MediaPipe model not found: {MODEL_PATH}")
        print("  Download: curl -o feature_extraction/tasks/face_landmarker.task "
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
        return

    video_dir = (os.path.join(DATA_DIR, "Videos")
                 if os.path.exists(os.path.join(DATA_DIR, "Videos"))
                 else os.path.join(DATA_DIR, split, "videos"))
    output_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(output_dir, exist_ok=True)

    mp4_files = sorted(Path(video_dir).glob("**/*.mp4"))
    if not mp4_files:
        print(f"[face_feature_extractor] No .mp4 files found in {video_dir}")
        return

    print(f"[face_feature_extractor] Found {len(mp4_files)} .mp4 files in '{split}' split")
    print(f"[face_feature_extractor] Feature dim: {N_SELECTED}×3 + {N_STRESS_AUS} AUs = "
          f"{N_SELECTED * 3 + N_STRESS_AUS} per frame")

    success, skip, fail = 0, 0, 0

    for mp4_path in tqdm(mp4_files, desc=f"  [{split}] Face Landmarks + AUs"):
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
            features = extract_face_features(str(mp4_path))
            np.save(out_path, features)
            success += 1
        except Exception as e:
            print(f"  [WARN] Failed: {mp4_path.name} — {e}")
            fail += 1

    print(f"[face_feature_extractor] Done — success: {success} | skipped: {skip} | failed: {fail}")
    print(f"[face_feature_extractor] Output saved to: {output_dir}")
    print(f"[face_feature_extractor] Shape per file: ({MAX_FRAMES}, {N_SELECTED * 3 + N_STRESS_AUS})")


# entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    args = parser.parse_args()

    if args.data_dir:
        DATA_DIR = args.data_dir
    process_all(split=args.split)
