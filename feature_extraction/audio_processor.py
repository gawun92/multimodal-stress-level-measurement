import os
import sys
import argparse
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm


SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
MAX_FRAMES = 1876
WINDOW_SEC = 10
HOP_SEC = 5
WINDOW_SAMP = WINDOW_SEC * SAMPLE_RATE
HOP_SAMP = HOP_SEC * SAMPLE_RATE
WINDOW_FRAMES = int(np.ceil(WINDOW_SAMP / HOP_LENGTH))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARC_DATA_DIR = "/project2/msoleyma_1026/group_14/data/stressid"
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "data", "stressid")
DATA_DIR = CARC_DATA_DIR if os.path.exists(CARC_DATA_DIR) else LOCAL_DATA_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "feature_extraction", "results", "mel_spectrograms")


def load_audio(wav_path: str) -> np.ndarray:
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    return audio


def compute_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def pad_or_truncate(mel: np.ndarray, max_frames: int = WINDOW_FRAMES) -> np.ndarray:
    T = mel.shape[1]
    if T >= max_frames:
        return mel[:, :max_frames]
    return np.pad(mel, ((0, 0), (0, max_frames - T)), mode="constant", constant_values=0.0)


def normalize(mel: np.ndarray) -> np.ndarray:
    mean = mel.mean()
    std = mel.std() + 1e-8
    return (mel - mean) / std


def process_single_file(wav_path: str) -> list[np.ndarray]:
    audio = load_audio(wav_path)
    mel = compute_mel_spectrogram(audio)
    mel = normalize(mel)

    T = mel.shape[1]
    step = WINDOW_FRAMES - OVERLAP_FRAMES

    windows = []
    for start in range(0, max(1, T - WINDOW_FRAMES + 1), step):
        window = mel[:, start:start + WINDOW_FRAMES]
        if window.shape[1] < WINDOW_FRAMES:
            window = pad_or_truncate(window, max_frames=WINDOW_FRAMES)
        window = window[np.newaxis, :]
        windows.append(window)

    return windows


def extract_windows(audio: np.ndarray) -> list:
    windows = []
    n = len(audio)

    if n < WINDOW_SAMP:
        chunk = np.pad(audio, (0, WINDOW_SAMP - n))
        mel = librosa.feature.melspectrogram(
            y=chunk, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        mel = mel[:, :WINDOW_FRAMES]
        if mel.shape[1] < WINDOW_FRAMES:
            mel = np.pad(mel, ((0, 0), (0, WINDOW_FRAMES - mel.shape[1])))
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        windows.append(mel[np.newaxis, :])
        return windows

    start = 0
    while start + WINDOW_SAMP <= n:
        chunk = audio[start: start + WINDOW_SAMP]
        mel = librosa.feature.melspectrogram(
            y=chunk, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        mel = mel[:, :WINDOW_FRAMES]
        if mel.shape[1] < WINDOW_FRAMES:
            mel = np.pad(mel, ((0, 0), (0, WINDOW_FRAMES - mel.shape[1])))
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        windows.append(mel[np.newaxis, :])
        start += HOP_SAMP

    return windows


def process_all_windowed(split: str = "train"):
    audio_dir = os.path.join(DATA_DIR, "Audio") if os.path.exists(os.path.join(DATA_DIR, "Audio")) else os.path.join(DATA_DIR, split, "audio")
    output_dir = os.path.join(
        BASE_DIR, "feature_extraction", "results", "mel_spectrograms_windowed", split
    )
    os.makedirs(output_dir, exist_ok=True)

    wav_files = sorted(Path(audio_dir).glob("**/*.wav"))
    if not wav_files:
        print(f"[audio_processor] No .wav files found in {audio_dir}")
        return

    print(f"[audio_processor] Windowed extraction — {len(wav_files)} files | "
          f"window={WINDOW_SEC}s hop={HOP_SEC}s → {WINDOW_FRAMES} frames/window")

    success, skip, fail, total_windows = 0, 0, 0, 0

    for wav_path in tqdm(wav_files, desc=f"  [{split}] Windowed Mel"):
        stem = wav_path.stem
        subject_id = stem.split("_")[0]
        task = "_".join(stem.split("_")[1:])

        subject_out_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_out_dir, exist_ok=True)

        existing = list(Path(subject_out_dir).glob(f"{task}_mel_w*.npy"))
        if existing:
            skip += 1
            total_windows += len(existing)
            continue

        try:
            audio = load_audio(str(wav_path))
            windows = extract_windows(audio)
            for i, w in enumerate(windows):
                out_path = os.path.join(subject_out_dir, f"{task}_mel_w{i:03d}.npy")
                np.save(out_path, w)
            success += 1
            total_windows += len(windows)
        except Exception as e:
            print(f"  [WARN] Failed: {wav_path.name} — {e}")
            fail += 1

    print(f"[audio_processor] Done — success: {success} | skipped: {skip} | failed: {fail}")
    print(f"[audio_processor] Total windows saved: {total_windows}")
    print(f"[audio_processor] Output: {output_dir}")


def process_iemocap():
    sys.path.insert(0, os.path.dirname(BASE_DIR))
    import config as cfg

    audio_dir = cfg.IEMOCAP_AUDIO_DIR
    output_dir = cfg.MEL_IEMOCAP_DIR
    os.makedirs(output_dir, exist_ok=True)

    wav_files = sorted(Path(audio_dir).glob("*.wav"))
    if not wav_files:
        print(f"[audio_processor] No .wav files found in {audio_dir}")
        return

    print(f"[audio_processor] IEMOCAP extraction — {len(wav_files)} clips → padded to {WINDOW_FRAMES} frames")

    success, skip, fail = 0, 0, 0
    for wav_path in tqdm(wav_files, desc="  [iemocap] Mel"):
        out_path = os.path.join(output_dir, f"{wav_path.stem}.npy")
        if os.path.exists(out_path):
            skip += 1
            continue
        try:
            audio = load_audio(str(wav_path))
            mel = librosa.feature.melspectrogram(
                y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
            )
            mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)
            if mel.shape[1] >= WINDOW_FRAMES:
                mel = mel[:, :WINDOW_FRAMES]
            else:
                mel = np.pad(mel, ((0, 0), (0, WINDOW_FRAMES - mel.shape[1])))
            mel = mel[np.newaxis, :]
            np.save(out_path, mel)
            success += 1
        except Exception as e:
            print(f"  [WARN] Failed: {wav_path.name} — {e}")
            fail += 1

    print(f"[audio_processor] IEMOCAP done — success: {success} | skipped: {skip} | failed: {fail}")
    print(f"[audio_processor] Output: {output_dir}")


def process_all(split: str = "train"):
    audio_dir = os.path.join(DATA_DIR, "Audio") if os.path.exists(os.path.join(DATA_DIR, "Audio")) else os.path.join(DATA_DIR, split, "audio")
    output_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(output_dir, exist_ok=True)

    wav_files = sorted(Path(audio_dir).glob("**/*.wav"))
    if not wav_files:
        print(f"[audio_processor] No .wav files found in {audio_dir}")
        return

    print(f"[audio_processor] Found {len(wav_files)} .wav files in '{split}' split")

    success, skip, fail = 0, 0, 0

    for wav_path in tqdm(wav_files, desc=f"  [{split}] Mel Spectrogram"):
        stem = wav_path.stem
        subject_id = stem.split("_")[0]
        task = "_".join(stem.split("_")[1:])

        subject_out_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_out_dir, exist_ok=True)
        first_out_path = os.path.join(subject_out_dir, f"{task}_mel_000.npy")

        if os.path.exists(first_out_path):
            skip += 1
            continue

        try:
            mels = process_single_file(str(wav_path))
            for i, mel_win in enumerate(mels):
                out_path = os.path.join(subject_out_dir, f"{task}_mel_{i:03d}.npy")
                np.save(out_path, mel_win)
            success += 1
        except Exception as e:
            print(f"  [WARN] Failed: {wav_path.name} — {e}")
            fail += 1

    print(f"[audio_processor] Done — success: {success} | skipped: {skip} | failed: {fail}")
    print(f"[audio_processor] Output saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Preprocessing: .wav → Mel Spectrogram → .npy")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--iemocap", action="store_true")
    args = parser.parse_args()

    if args.windowed:
        process_all_windowed(split=args.split)
    elif args.iemocap:
        process_iemocap()
    else:
        process_all(split=args.split)
