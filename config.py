import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARC_DATA_DIR = "/project2/msoleyma_1026/group_14/data/stressid"
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "data", "stressid")
DATA_DIR = CARC_DATA_DIR if os.path.exists(CARC_DATA_DIR) else LOCAL_DATA_DIR

LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
MEL_DIR = os.path.join(BASE_DIR, "feature_extraction", "results", "mel_spectrograms", "train")
MEL_WINDOWED_DIR = os.path.join(BASE_DIR, "feature_extraction", "results", "mel_spectrograms_windowed", "train")
MEL_IEMOCAP_DIR = os.path.join(BASE_DIR, "feature_extraction", "results", "mel_spectrograms_iemocap")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

IEMOCAP_AUDIO_DIR = os.path.expanduser("~/Downloads/processed/Audio")
IEMOCAP_LABELS_CSV = os.path.expanduser("~/Downloads/processed/labels.csv")

N_MELS = 128
MAX_FRAMES = 1876
WINDOW_FRAMES = 313
WINDOW_SEC = 10
HOP_SEC = 5

CNN_CHANNELS = [1, 32, 64, 128]
CNN_KERNEL_SIZE = 3
CNN_PADDING = 1

EMBED_DIM = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_FF_DIM = 256
TRANSFORMER_DROPOUT = 0.1

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 80
PATIENCE = 15

NUM_CLASSES_BINARY = 2
NUM_CLASSES_AFFECT3 = 3
LABEL_COLUMN = "binary-stress"

NUM_FOLDS = 5
RANDOM_SEED = 42
VAL_RATIO = 0.15
ENSEMBLE_SEEDS = [42, 123, 456]
TTA_STEPS = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

AUDIO_TASKS = ["Counting1", "Counting2", "Counting3", "Math", "Reading", "Speaking", "Stroop"]

HELD_OUT_SUBJECTS = ["wssm", "x1q3", "y8c3", "y9z6"]
