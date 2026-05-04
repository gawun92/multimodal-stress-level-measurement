import os
import torch


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARC_DATA_DIR = "/project2/msoleyma_1026/group_14/data/stressid"
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "data", "stressid")
_DESKTOP_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "StressID Dataset")
DATA_DIR = (CARC_DATA_DIR if os.path.exists(CARC_DATA_DIR)
            else _DESKTOP_DATA_DIR if os.path.exists(_DESKTOP_DATA_DIR)
            else LOCAL_DATA_DIR)

LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
AUDIO_WAV_DIR = os.path.join(DATA_DIR, "Audio")
MEL_DIR = os.path.join(BASE_DIR, "feature_extraction", "results", "mel_spectrograms", "train")
GESTURE_DIR = os.path.join(BASE_DIR, "feature_extraction", "results", "gesture", "train")
UPPER_BODY_DIR = os.path.join(BASE_DIR, "feature_extraction", "results", "upper_body", "train")
FACE_DIR = os.path.join(BASE_DIR, "feature_extraction", "results", "face", "train")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

AUDIO_CKPT_PATH = os.path.join(BASE_DIR, "New_features", "audio_features", "audio_branch_fold3_binary-stress.pt")
FACE_FEATURE_DIR = os.path.join(BASE_DIR, "New_features", "face_features")
GESTURE_FEATURE_DIR = os.path.join(BASE_DIR, "New_features", "gesture_features", "extracted_features")
GESTURE_CKPT_PATH = os.path.join(BASE_DIR, "New_features", "gesture_features", "gesture_branch_fold0_binary-stress_20260417_153259.pt")
GESTURE_CKPT_INPUT_DIM = 33


# Audio feature dimensions
N_MELS = 128
MAX_FRAMES = 1876


# Gesture feature dimensions
GESTURE_MAX_FRAMES = 300
GESTURE_N_LANDMARKS = 21
GESTURE_INPUT_DIM = GESTURE_N_LANDMARKS * 3


# Upper-body feature dimensions
UPPER_BODY_MAX_FRAMES = 300
UPPER_BODY_N_LANDMARKS = 11
UPPER_BODY_INPUT_DIM = UPPER_BODY_N_LANDMARKS * 3


# Face feature dimensions
FACE_MAX_FRAMES = 300
FACE_N_LANDMARKS = 92
FACE_AU_DIM = 10
FACE_INPUT_DIM = FACE_N_LANDMARKS * 3 + FACE_AU_DIM


# Model architecture
CNN_CHANNELS = [1, 32, 64, 128]
CNN_KERNEL_SIZE = 3
CNN_PADDING = 1

EMBED_DIM = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_FF_DIM = 256
TRANSFORMER_DROPOUT = 0.1


# Training
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 50
PATIENCE = 10

NUM_CLASSES_BINARY = 2
NUM_CLASSES_AFFECT3 = 3
LABEL_COLUMN = "binary-stress"


# Data split
NUM_FOLDS = 5
RANDOM_SEED = 42
VAL_RATIO = 0.15


# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# Task lists
AUDIO_TASKS = [
    "Counting1",
    "Counting2",
    "Counting3",
    "Math",
    "Reading",
    "Speaking",
    "Stroop",
]

VIDEO_TASKS = [
    "Baseline",
    "Breathing",
    "Counting1",
    "Counting2",
    "Counting3",
    "Math",
    "Reading",
    "Relax",
    "Speaking",
    "Stroop",
    "Video1",
    "Video2",
]


# Held-out test subjects
HELD_OUT_SUBJECTS = ["wssm", "x1q3", "y8c3", "y9z6"]