"""
dataset.py

PyTorch datasets for loading pre-extracted modality features with stress labels.
Supports subject-level train/val/test splitting via k-fold CV.
"""

import os

import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as T
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

import config


class StressAudioDataset(Dataset):
    """
    Loads pre-computed mel spectrograms and corresponding stress labels.

    Each sample:
        X: torch.FloatTensor of shape (1, 128, 1876)
        y: torch.LongTensor scalar
    """

    def __init__(
        self,
        subject_ids,
        label_col="binary-stress",
        mel_dir=None,
        labels_csv=None,
        augment=False,
    ):
        self.mel_dir = mel_dir or config.MEL_DIR
        self.labels_csv = labels_csv or config.LABELS_CSV
        self.label_col = label_col
        self.augment = augment

        if self.augment:
            self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
            self.time_mask = T.TimeMasking(time_mask_param=35)

        labels_df = pd.read_csv(self.labels_csv).set_index("subject/task")

        self.samples = []
        for subject_id in subject_ids:
            for task in config.AUDIO_TASKS:
                key = f"{subject_id}_{task}"
                npy_path = os.path.join(self.mel_dir, subject_id, f"{task}_mel.npy")

                if key not in labels_df.index:
                    continue
                if not os.path.exists(npy_path):
                    continue

                label = int(labels_df.loc[key, self.label_col])
                self.samples.append((npy_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        mel = np.load(npy_path)
        X = torch.from_numpy(mel).float()

        if self.augment:
            for _ in range(np.random.randint(1, 3)):
                X = self.freq_mask(X)
            for _ in range(np.random.randint(1, 3)):
                X = self.time_mask(X)

        y = torch.tensor(label, dtype=torch.long)
        return X, y


class StressGestureDataset(Dataset):
    """
    Loads pre-extracted hand-landmark .npy files and stress labels.

    Each sample:
        X: torch.FloatTensor of shape (T, 63)
        mask: torch.BoolTensor of shape (T,)
        y: torch.LongTensor scalar
    """

    def __init__(
        self,
        subject_ids,
        label_col="binary-stress",
        gesture_dir=None,
        labels_csv=None,
        tasks=None,
        return_mask=False,
        require_mask=False,
        window_len=None,
        min_valid_frames=1,
        window_mode="best",
    ):
        self.gesture_dir = gesture_dir or config.GESTURE_DIR
        self.labels_csv = labels_csv or config.LABELS_CSV
        self.label_col = label_col
        self.tasks = tasks or config.VIDEO_TASKS
        self.return_mask = return_mask
        self.require_mask = require_mask
        self.window_len = window_len
        self.min_valid_frames = min_valid_frames
        self.window_mode = window_mode

        if self.window_mode not in {"best"}:
            raise ValueError(
                f"Unsupported window_mode='{self.window_mode}'. Supported: 'best'."
            )
        if self.window_len is not None and self.window_len <= 0:
            raise ValueError("window_len must be positive when provided.")
        if self.min_valid_frames < 0:
            raise ValueError("min_valid_frames must be >= 0.")

        labels_df = pd.read_csv(self.labels_csv).set_index("subject/task")

        self.samples = []
        for subject_id in subject_ids:
            for task in self.tasks:
                key = f"{subject_id}_{task}"
                npy_path = os.path.join(self.gesture_dir, subject_id, f"{task}_gesture.npy")
                mask_path = os.path.join(self.gesture_dir, subject_id, f"{task}_gesture_mask.npy")

                if key not in labels_df.index:
                    continue
                if not os.path.exists(npy_path):
                    continue
                if self.require_mask and not os.path.exists(mask_path):
                    continue

                label = int(labels_df.loc[key, self.label_col])
                start = None
                end = None
                if self.window_len is not None:
                    if not os.path.exists(mask_path):
                        continue
                    mask_np = np.load(mask_path).astype(bool)
                    start, end, valid_count = self._select_window(mask_np)
                    if valid_count < self.min_valid_frames:
                        continue
                self.samples.append((npy_path, mask_path, label, start, end))

    def __len__(self):
        return len(self.samples)

    def _select_window(self, mask):
        total_frames = int(mask.shape[0])
        if total_frames == 0:
            return 0, 0, 0

        if self.window_len is None or self.window_len >= total_frames:
            valid_count = int(mask.sum())
            return 0, total_frames, valid_count

        window = np.ones(self.window_len, dtype=np.int32)
        scores = np.convolve(mask.astype(np.int32), window, mode="valid")
        best_start = int(scores.argmax())
        best_count = int(scores[best_start])
        return best_start, best_start + self.window_len, best_count

    def __getitem__(self, idx):
        npy_path, mask_path, label, start, end = self.samples[idx]
        landmarks = np.load(npy_path)
        T_frames, n_landmarks, coord_dim = landmarks.shape
        X = torch.from_numpy(landmarks.reshape(T_frames, n_landmarks * coord_dim)).float()

        if os.path.exists(mask_path):
            mask = torch.from_numpy(np.load(mask_path).astype(bool))
        else:
            mask = torch.any(X != 0, dim=1)

        if start is not None:
            X = X[start:end]
            mask = mask[start:end]

        y = torch.tensor(label, dtype=torch.long)
        if self.return_mask:
            return X, mask, y
        return X, y


class StressUpperBodyDataset(Dataset):
    """
    Loads pre-extracted upper-body .npy files and stress labels.

    Each sample:
        X: torch.FloatTensor of shape (T, 33)
        mask: torch.BoolTensor of shape (T,)
        y: torch.LongTensor scalar
    """

    def __init__(
        self,
        subject_ids,
        label_col="binary-stress",
        upper_body_dir=None,
        labels_csv=None,
        tasks=None,
        return_mask=False,
        require_mask=False,
        window_len=None,
        min_valid_frames=1,
        window_mode="best",
    ):
        self.upper_body_dir = upper_body_dir or config.UPPER_BODY_DIR
        self.labels_csv = labels_csv or config.LABELS_CSV
        self.label_col = label_col
        self.tasks = tasks or config.VIDEO_TASKS
        self.return_mask = return_mask
        self.require_mask = require_mask
        self.window_len = window_len
        self.min_valid_frames = min_valid_frames
        self.window_mode = window_mode

        if self.window_mode not in {"best"}:
            raise ValueError(
                f"Unsupported window_mode='{self.window_mode}'. Supported: 'best'."
            )
        if self.window_len is not None and self.window_len <= 0:
            raise ValueError("window_len must be positive when provided.")
        if self.min_valid_frames < 0:
            raise ValueError("min_valid_frames must be >= 0.")

        labels_df = pd.read_csv(self.labels_csv).set_index("subject/task")

        self.samples = []
        for subject_id in subject_ids:
            for task in self.tasks:
                key = f"{subject_id}_{task}"
                npy_path = os.path.join(
                    self.upper_body_dir, subject_id, f"{task}_upperbody.npy"
                )
                mask_path = os.path.join(
                    self.upper_body_dir, subject_id, f"{task}_upperbody_mask.npy"
                )

                if key not in labels_df.index:
                    continue
                if not os.path.exists(npy_path):
                    continue
                if self.require_mask and not os.path.exists(mask_path):
                    continue

                label = int(labels_df.loc[key, self.label_col])
                start = None
                end = None
                if self.window_len is not None:
                    if not os.path.exists(mask_path):
                        continue
                    mask_np = np.load(mask_path).astype(bool)
                    start, end, valid_count = self._select_window(mask_np)
                    if valid_count < self.min_valid_frames:
                        continue
                self.samples.append((npy_path, mask_path, label, start, end))

    def __len__(self):
        return len(self.samples)

    def _select_window(self, mask):
        total_frames = int(mask.shape[0])
        if total_frames == 0:
            return 0, 0, 0

        if self.window_len is None or self.window_len >= total_frames:
            valid_count = int(mask.sum())
            return 0, total_frames, valid_count

        window = np.ones(self.window_len, dtype=np.int32)
        scores = np.convolve(mask.astype(np.int32), window, mode="valid")
        best_start = int(scores.argmax())
        best_count = int(scores[best_start])
        return best_start, best_start + self.window_len, best_count

    def __getitem__(self, idx):
        npy_path, mask_path, label, start, end = self.samples[idx]
        landmarks = np.load(npy_path)
        T_frames, n_landmarks, coord_dim = landmarks.shape
        X = torch.from_numpy(landmarks.reshape(T_frames, n_landmarks * coord_dim)).float()

        if os.path.exists(mask_path):
            mask = torch.from_numpy(np.load(mask_path).astype(bool))
        else:
            mask = torch.any(X != 0, dim=1)

        if start is not None:
            X = X[start:end]
            mask = mask[start:end]

        y = torch.tensor(label, dtype=torch.long)
        if self.return_mask:
            return X, mask, y
        return X, y


class StressFaceDataset(Dataset):
    """
    Loads pre-extracted face .npy files and stress labels.

    Supports both:
        - flattened per-frame feature arrays of shape (T, F)
        - landmark tensors of shape (T, L, C)
    """

    def __init__(
        self,
        subject_ids,
        label_col="binary-stress",
        face_dir=None,
        labels_csv=None,
        tasks=None,
    ):
        self.face_dir = face_dir or config.FACE_DIR
        self.labels_csv = labels_csv or config.LABELS_CSV
        self.label_col = label_col
        self.tasks = tasks or config.VIDEO_TASKS

        labels_df = pd.read_csv(self.labels_csv).set_index("subject/task")

        self.samples = []
        for subject_id in subject_ids:
            for task in self.tasks:
                key = f"{subject_id}_{task}"
                npy_path = os.path.join(self.face_dir, subject_id, f"{task}_face.npy")

                if key not in labels_df.index:
                    continue
                if not os.path.exists(npy_path):
                    continue

                label = int(labels_df.loc[key, self.label_col])
                self.samples.append((npy_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        arr = np.load(npy_path)
        if arr.ndim == 3:
            T_frames, n_landmarks, coord_dim = arr.shape
            X = torch.from_numpy(arr.reshape(T_frames, n_landmarks * coord_dim)).float()
        elif arr.ndim == 2:
            X = torch.from_numpy(arr).float()
        else:
            raise ValueError(f"Unsupported face feature shape {arr.shape} in {npy_path}")

        y = torch.tensor(label, dtype=torch.long)
        return X, y


def _get_all_subjects(feature_dir):
    if not os.path.exists(feature_dir):
        return []
    return sorted(
        d for d in os.listdir(feature_dir) if os.path.isdir(os.path.join(feature_dir, d))
    )


def get_all_audio_subjects(mel_dir=None):
    return _get_all_subjects(mel_dir or config.MEL_DIR)


def get_all_gesture_subjects(gesture_dir=None):
    return _get_all_subjects(gesture_dir or config.GESTURE_DIR)


def get_all_upper_body_subjects(upper_body_dir=None):
    return _get_all_subjects(upper_body_dir or config.UPPER_BODY_DIR)


def get_all_face_subjects(face_dir=None):
    return _get_all_subjects(face_dir or config.FACE_DIR)


def get_held_out_subjects(mel_dir=None):
    mel_dir = mel_dir or config.MEL_DIR
    return [
        subject_id
        for subject_id in config.HELD_OUT_SUBJECTS
        if os.path.isdir(os.path.join(mel_dir, subject_id))
    ]


def get_subject_splits(
    fold=0,
    n_folds=None,
    seed=None,
    val_ratio=None,
    subject_fn=None,
    tasks=None,
):
    """
    Subject-level stratified k-fold split on the CV pool.

    Args:
        subject_fn: callable returning available subject ids for the target modality.
        tasks: task list to use when computing subject-level stratification labels.
    """

    n_folds = n_folds or config.NUM_FOLDS
    seed = seed or config.RANDOM_SEED
    val_ratio = val_ratio or config.VAL_RATIO
    subject_fn = subject_fn or get_all_audio_subjects
    tasks = tasks or config.AUDIO_TASKS

    all_subjects = subject_fn()
    if not all_subjects:
        raise RuntimeError("No subjects found. Run feature extraction first.")

    held_out_set = set(config.HELD_OUT_SUBJECTS)
    subjects = np.array([subject_id for subject_id in all_subjects if subject_id not in held_out_set])

    labels_df = pd.read_csv(config.LABELS_CSV).set_index("subject/task")
    subject_labels = []
    for subject_id in subjects:
        task_labels = []
        for task in tasks:
            key = f"{subject_id}_{task}"
            if key in labels_df.index:
                task_labels.append(int(labels_df.loc[key, "binary-stress"]))
        dominant = int(np.round(np.mean(task_labels))) if task_labels else 1
        subject_labels.append(dominant)
    subject_labels = np.array(subject_labels)

    if fold < 0 or fold >= n_folds:
        raise ValueError(
            f"fold={fold} is out of range for n_folds={n_folds}. "
            f"Valid fold indices are 0-{n_folds - 1}."
        )

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
    val_idx = np.concatenate(
        [stressed_idx[:n_val_stressed], no_stress_idx[:n_val_no_stress]]
    )
    train_idx = np.concatenate(
        [stressed_idx[n_val_stressed:], no_stress_idx[n_val_no_stress:]]
    )

    val_subjects = train_val_subjects[val_idx].tolist()
    train_subjects = train_val_subjects[train_idx].tolist()
    return train_subjects, val_subjects, test_subjects
