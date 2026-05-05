"""
iemocap_processor.py

Extracts Mel spectrograms from the downloaded IEMOCAP audio dataset
for pre-training the Audio Branch.

Input: ~/Downloads/processed/Audio/*.wav
Output: feature_extraction/results/mel_spectrograms_iemocap/*.npy
Shape: (1, 128, 1876), aligned with StressID full-clip format
"""

import os
import argparse
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Config
SAMPLE_RATE = 16000
N_MELS      = 128
N_FFT       = 1024
HOP_LENGTH  = 512
MAX_FRAMES  = 1876  # ~60s clip, align with full-clip StressID padding

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(BASE_DIR, "feature_extraction", "results", "mel_spectrograms_iemocap")

# Default teammate IEMOCAP locations
IEMOCAP_LABELS_CSV = os.path.expanduser("~/Downloads/processed/labels.csv")
IEMOCAP_AUDIO_DIR  = os.path.expanduser("~/Downloads/processed/Audio")


def load_audio(wav_path: str) -> np.ndarray:
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    return audio

def compute_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)

def pad_or_truncate(mel: np.ndarray) -> np.ndarray:
    """Pad to MAX_FRAMES (1876) to match the StressID full-clip dimension."""
    T = mel.shape[1]
    if T >= MAX_FRAMES:
        return mel[:, :MAX_FRAMES]
    return np.pad(mel, ((0, 0), (0, MAX_FRAMES - T)), mode="constant", constant_values=0.0)

def normalize(mel: np.ndarray) -> np.ndarray:
    mean = mel.mean()
    std  = mel.std() + 1e-8
    return (mel - mean) / std

def process_single_file(wav_path: str) -> np.ndarray:
    audio = load_audio(wav_path)
    mel   = compute_mel_spectrogram(audio)
    mel   = pad_or_truncate(mel)
    mel   = normalize(mel)
    mel   = mel[np.newaxis, :]  # (1, 128, 1876)
    return mel

def extract_all():
    if not os.path.exists(IEMOCAP_LABELS_CSV) or not os.path.exists(IEMOCAP_AUDIO_DIR):
        print(f"[iemocap_processor] ERROR: IEMOCAP data not found at expected path:\n  {IEMOCAP_AUDIO_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load labels to only process mapped clips
    df = pd.read_csv(IEMOCAP_LABELS_CSV)
    mapped_clips = set(df['subject/task'] + ".wav")
    
    wav_files = [Path(os.path.join(IEMOCAP_AUDIO_DIR, f)) for f in os.listdir(IEMOCAP_AUDIO_DIR) if f in mapped_clips]
    
    print(f"[iemocap_processor] Found {len(wav_files)} valid mapped .wav files.")
    
    success, skip, fail = 0, 0, 0
    for wav_path in tqdm(wav_files, desc="IEMOCAP Mel Extraction"):
        stem = wav_path.stem
        out_path = os.path.join(OUTPUT_DIR, f"{stem}_mel.npy")
        
        if os.path.exists(out_path):
            skip += 1
            continue
            
        try:
            mel = process_single_file(str(wav_path))
            np.save(out_path, mel)
            success += 1
        except Exception as e:
            print(f"  [WARN] Failed: {wav_path.name}:{e}")
            fail += 1

    print(f"[iemocap_processor] Done:success: {success} | skipped: {skip} | failed: {fail}")
    print(f"[iemocap_processor] Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Mel Spectrograms from IEMOCAP")
    args = parser.parse_args()
    extract_all()
