"""
audio_processor.py

Stage 1 - Audio Preprocessing
Converts raw .wav files into Mel Spectrogram and saves as .npy

Input  : data/{split}/audio/{subjectID}/{subjectID}_{task}.wav
Output : feature_extraction/results/mel_spectrograms/{split}/{subjectID}/{task}_mel.npy
         Shape: (1, N_MELS, T) — (channel, mel_bins, time_frames)

Usage:
    python feature_extraction/audio_processor.py --split train
    python feature_extraction/audio_processor.py --split test
"""

import os
import argparse
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
SAMPLE_RATE = 16000       # Resample all audio to 16kHz
N_MELS      = 128         # Number of Mel filterbanks
N_FFT       = 1024        # FFT window size
HOP_LENGTH  = 512         # Hop length between frames
MAX_FRAMES  = 1876        # Max time frames (≈ 60s at 16kHz / hop 512)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARC_DATA_DIR = "/project2/msoleyma_1026/group_14/data/stressid"
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "data", "stressid")
DATA_DIR    = CARC_DATA_DIR if os.path.exists(CARC_DATA_DIR) else LOCAL_DATA_DIR
OUTPUT_DIR  = os.path.join(BASE_DIR, "feature_extraction", "results", "mel_spectrograms")


# ─────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────
def load_audio(wav_path: str) -> np.ndarray:
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    return audio


def compute_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def pad_or_truncate(mel: np.ndarray, max_frames: int = MAX_FRAMES) -> np.ndarray:
    T = mel.shape[1]
    if T >= max_frames:
        return mel[:, :max_frames]
    return np.pad(mel, ((0, 0), (0, max_frames - T)), mode="constant", constant_values=0.0)


def normalize(mel: np.ndarray) -> np.ndarray:
    mean = mel.mean()
    std  = mel.std() + 1e-8
    return (mel - mean) / std


def process_single_file(wav_path: str) -> np.ndarray:
    audio = load_audio(wav_path)
    mel   = compute_mel_spectrogram(audio)
    mel   = pad_or_truncate(mel)
    mel   = normalize(mel)
    mel   = mel[np.newaxis, :]      # (1, N_MELS, T)
    return mel


# ─────────────────────────────────────────
# Batch Processing
# ─────────────────────────────────────────
def process_all(split: str = "train"):
    audio_dir  = os.path.join(DATA_DIR, "Audio", "bfl5") if os.path.exists(os.path.join(DATA_DIR, "Audio")) else os.path.join(DATA_DIR, split, "audio")
    output_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(output_dir, exist_ok=True)

    wav_files = sorted(Path(audio_dir).glob("**/*.wav"))
    if not wav_files:
        print(f"[audio_processor] No .wav files found in {audio_dir}")
        return

    print(f"[audio_processor] Found {len(wav_files)} .wav files in '{split}' split")

    success, skip, fail = 0, 0, 0

    for wav_path in tqdm(wav_files, desc=f"  [{split}] Mel Spectrogram"):
        stem       = wav_path.stem                    # "2ea4_Counting1"
        subject_id = stem.split("_")[0]               # "2ea4"
        task       = "_".join(stem.split("_")[1:])    # "Counting1"

        subject_out_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_out_dir, exist_ok=True)
        out_path = os.path.join(subject_out_dir, f"{task}_mel.npy")

        if os.path.exists(out_path):
            skip += 1
            continue

        try:
            mel = process_single_file(str(wav_path))
            np.save(out_path, mel)
            success += 1
        except Exception as e:
            print(f"  [WARN] Failed: {wav_path.name} — {e}")
            fail += 1

    print(f"[audio_processor] Done — success: {success} | skipped: {skip} | failed: {fail}")
    print(f"[audio_processor] Output saved to: {output_dir}")


# ─────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Preprocessing: .wav → Mel Spectrogram → .npy")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                        help="Which data split to process (default: train)")
    args = parser.parse_args()

    process_all(split=args.split)