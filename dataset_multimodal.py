import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

import config

_SR = 16000
_N_MELS = config.N_MELS
_N_FFT = 1024
_HOP = 512
_MAX_FRAMES = config.MAX_FRAMES


def _wav_to_mel(wav_path: str) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(wav_path, sr=_SR, mono=True)
    mel = librosa.feature.melspectrogram(y=audio, sr=_SR, n_mels=_N_MELS,
                                          n_fft=_N_FFT, hop_length=_HOP)
    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    T = mel.shape[1]
    if T >= _MAX_FRAMES:
        mel = mel[:, :_MAX_FRAMES]
    else:
        mel = np.pad(mel, ((0, 0), (0, _MAX_FRAMES - T)), constant_values=0.0)
    mean, std = mel.mean(), mel.std() + 1e-8
    mel = (mel - mean) / std
    return mel[np.newaxis]


def _pad_or_truncate_face(arr: np.ndarray, max_frames: int = 300) -> np.ndarray:
    arr = arr.reshape(arr.shape[0], -1).astype(np.float32)
    T, F = arr.shape
    if T >= max_frames:
        return arr[:max_frames]
    return np.concatenate([arr, np.zeros((max_frames - T, F), dtype=np.float32)], axis=0)


def _pad_or_truncate_gesture(arr: np.ndarray, max_frames: int = 300) -> np.ndarray:
    arr = arr.reshape(arr.shape[0], -1).astype(np.float32)
    T, F = arr.shape
    if T >= max_frames:
        return arr[:max_frames]
    return np.concatenate([arr, np.zeros((max_frames - T, F), dtype=np.float32)], axis=0)


def _has_gesture_features(gesture_dir: str) -> bool:
    if not os.path.isdir(gesture_dir):
        return False
    for entry in os.listdir(gesture_dir):
        subdir = os.path.join(gesture_dir, entry)
        if os.path.isdir(subdir):
            for f in os.listdir(subdir):
                if f.endswith("_gesture.npy"):
                    return True
    return False


class MultimodalStressDataset(Dataset):
    def __init__(self, subject_ids, label_col="binary-stress",
                 wav_dir=None, face_dir=None, gesture_dir=None, mel_dir=None,
                 labels_csv=None, preload=True):
        self.wav_dir  = wav_dir  or config.AUDIO_WAV_DIR
        self.face_dir = face_dir or config.FACE_FEATURE_DIR
        self.mel_dir  = mel_dir  or config.MEL_DIR
        self.gesture_dir = gesture_dir or config.GESTURE_FEATURE_DIR
        self.label_col = label_col
        self.preload = preload
        self.use_gesture = _has_gesture_features(self.gesture_dir)

        labels_df = pd.read_csv(labels_csv or config.LABELS_CSV)
        labels_df = labels_df.set_index("subject/task")

        self.samples = []

        for sid in subject_ids:
            for task in config.AUDIO_TASKS:
                key = f"{sid}_{task}"
                mel_path     = os.path.join(self.mel_dir,  sid, f"{task}_mel.npy")
                wav_path     = os.path.join(self.wav_dir,  sid, f"{sid}_{task}.wav")
                face_path    = os.path.join(self.face_dir, sid, f"{task}_face.npy")
                gesture_path = os.path.join(self.gesture_dir, sid, f"{task}_gesture.npy")

                if os.path.exists(mel_path):
                    audio_src, use_mel = mel_path, True
                elif os.path.exists(wav_path):
                    audio_src, use_mel = wav_path, False
                else:
                    continue

                if key not in labels_df.index:
                    continue
                if not os.path.exists(face_path):
                    continue
                if self.use_gesture and not os.path.exists(gesture_path):
                    continue

                label = int(labels_df.loc[key, self.label_col])
                gpath = gesture_path if self.use_gesture else None
                self.samples.append((audio_src, face_path, gpath, label, use_mel))

        if self.preload:
            self._cache = []
            for audio_src, face_path, gesture_path, label, use_mel in self.samples:
                mel  = np.load(audio_src) if use_mel else _wav_to_mel(audio_src)
                face = _pad_or_truncate_face(np.load(face_path))
                gest = _pad_or_truncate_gesture(np.load(gesture_path)) if gesture_path else None
                self._cache.append((mel, face, gest, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preload:
            mel, face, gest, label = self._cache[idx]
        else:
            audio_src, face_path, gesture_path, label, use_mel = self.samples[idx]
            mel  = np.load(audio_src) if use_mel else _wav_to_mel(audio_src)
            face = _pad_or_truncate_face(np.load(face_path))
            gest = _pad_or_truncate_gesture(np.load(gesture_path)) if gesture_path else None

        if self.use_gesture:
            return (
                torch.from_numpy(mel).float(),
                torch.from_numpy(face).float(),
                torch.from_numpy(gest).float(),
                torch.tensor(label, dtype=torch.long),
            )
        return (
            torch.from_numpy(mel).float(),
            torch.from_numpy(face).float(),
            torch.tensor(label, dtype=torch.long),
        )


def _get_face_subjects(face_dir=None):
    face_dir = face_dir or config.FACE_FEATURE_DIR
    return sorted([
        d for d in os.listdir(face_dir)
        if os.path.isdir(os.path.join(face_dir, d)) and not d.startswith(".")
    ])


def get_multimodal_subject_splits(fold=0, n_folds=None, seed=None, val_ratio=None,
                                   label_col="binary-stress"):
    n_folds = n_folds or config.NUM_FOLDS
    seed = seed or config.RANDOM_SEED
    val_ratio = val_ratio or config.VAL_RATIO

    all_subjects = _get_face_subjects()
    held_out_set = set(config.HELD_OUT_SUBJECTS)
    subjects = np.array([s for s in all_subjects if s not in held_out_set])

    labels_df = pd.read_csv(config.LABELS_CSV).set_index("subject/task")
    subject_labels = []
    for s in subjects:
        task_labels = [
            int(labels_df.loc[f"{s}_{t}", label_col])
            for t in config.AUDIO_TASKS
            if f"{s}_{t}" in labels_df.index
        ]
        dominant = int(np.round(np.mean(task_labels))) if task_labels else 1
        subject_labels.append(dominant)
    subject_labels = np.array(subject_labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(subjects, subject_labels))
    train_val_idx, test_idx = splits[fold]

    test_subjects = subjects[test_idx].tolist()
    train_val_subjects = subjects[train_val_idx]
    train_val_labels = subject_labels[train_val_idx]

    np.random.seed(seed + fold)
    n_val = max(1, int(len(train_val_subjects) * val_ratio))
    stressed_idx = np.where(train_val_labels == 1)[0]
    no_stress_idx = np.where(train_val_labels == 0)[0]
    np.random.shuffle(stressed_idx)
    np.random.shuffle(no_stress_idx)
    n_val_stressed = max(1, round(n_val * len(stressed_idx) / len(train_val_subjects)))
    n_val_no_stress = max(0, n_val - n_val_stressed)
    val_idx = np.concatenate([stressed_idx[:n_val_stressed], no_stress_idx[:n_val_no_stress]])
    train_idx = np.concatenate([stressed_idx[n_val_stressed:], no_stress_idx[n_val_no_stress:]])

    val_subjects = train_val_subjects[val_idx].tolist()
    train_subjects = train_val_subjects[train_idx].tolist()
    return train_subjects, val_subjects, test_subjects


def get_multimodal_held_out_subjects():
    face_subjects = set(_get_face_subjects())
    return [s for s in config.HELD_OUT_SUBJECTS if s in face_subjects]
