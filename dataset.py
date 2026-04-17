"""
dataset.py

PyTorch Dataset for loading mel spectrogram .npy files with stress labels.
Supports subject-level train/val/test splitting via k-fold CV.

Usage:
    from dataset import StressAudioDataset, get_subject_splits
    train_subjects, val_subjects, test_subjects = get_subject_splits(fold=0)
    train_ds = StressAudioDataset(train_subjects, label_col="binary-stress")
"""

import os
import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as T
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, StratifiedKFold

import config


class StressAudioDataset(Dataset):
    """
    Loads pre-computed mel spectrograms and corresponding stress labels.

    Each sample:
        X: torch.FloatTensor of shape (1, 128, 1876)
        y: torch.LongTensor scalar (0 or 1 for binary, 0/1/2 for affect3)
    """

    def __init__(self, subject_ids, label_col="binary-stress", mel_dir=None, labels_csv=None,
                 augment=False, windowed=False):
        self.windowed = windowed
        # Use windowed mel dir if requested and it exists, else fall back to full-clip
        if windowed and os.path.exists(config.MEL_WINDOWED_DIR):
            self.mel_dir = mel_dir or config.MEL_WINDOWED_DIR
        else:
            if windowed:
                print("[dataset] Windowed mels not found — falling back to full-clip. "
                      "Run: python feature_extraction/audio_processor.py --windowed")
            self.mel_dir = mel_dir or config.MEL_DIR
            self.windowed = False

        self.labels_csv = labels_csv or config.LABELS_CSV
        self.label_col = label_col
        self.augment = augment

        # SpecAugment transforms
        if self.augment:
            self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
            self.time_mask = T.TimeMasking(time_mask_param=35)

        # Load labels
        labels_df = pd.read_csv(self.labels_csv)
        labels_df = labels_df.set_index("subject/task")

        # Build sample list: (npy_path, label)
        # Windowed mode: discovers all _w000, _w001, ... files per subject/task
        # Full-clip mode: one file per subject/task as before
        self.samples = []
        for subject_id in subject_ids:
            for task in config.AUDIO_TASKS:
                key = f"{subject_id}_{task}"
                if key not in labels_df.index:
                    continue

                label = int(labels_df.loc[key, self.label_col])

                if self.windowed:
                    # Discover all windows for this clip
                    subject_dir = os.path.join(self.mel_dir, subject_id)
                    if not os.path.isdir(subject_dir):
                        continue
                    window_files = sorted(
                        Path(subject_dir).glob(f"{task}_mel_w*.npy")
                    )
                    for wf in window_files:
                        self.samples.append((str(wf), label))
                else:
                    npy_path = os.path.join(self.mel_dir, subject_id, f"{task}_mel.npy")
                    if not os.path.exists(npy_path):
                        continue
                    self.samples.append((npy_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        mel = np.load(npy_path)  # (1, 128, 1876)
        X = torch.from_numpy(mel).float()
        
        if self.augment:
            # Randomly apply 0-2 frequency masks and 0-2 time masks
            for _ in range(np.random.randint(1, 3)):
                X = self.freq_mask(X)
            for _ in range(np.random.randint(1, 3)):
                X = self.time_mask(X)

        y = torch.tensor(label, dtype=torch.long)
        return X, y


def get_all_audio_subjects(mel_dir=None):
    """Return sorted list of subject IDs that have extracted mel spectrograms."""
    mel_dir = mel_dir or config.MEL_DIR
    if not os.path.exists(mel_dir):
        return []
    subjects = sorted([
        d for d in os.listdir(mel_dir)
        if os.path.isdir(os.path.join(mel_dir, d))
    ])
    return subjects


def get_held_out_subjects(mel_dir=None):
    """
    Return the held-out subjects defined in config.HELD_OUT_SUBJECTS that
    actually have mel spectrograms on disk.  These subjects are NEVER included
    in any training or validation split — only in final held-out evaluation.
    """
    mel_dir = mel_dir or config.MEL_DIR
    return [s for s in config.HELD_OUT_SUBJECTS
            if os.path.isdir(os.path.join(mel_dir, s))]


def get_subject_splits(fold=0, n_folds=None, seed=None, val_ratio=None):
    """
    Subject-level stratified k-fold split on the CV pool (held-out subjects excluded).
    Returns (train_subjects, val_subjects, test_subjects).
    The val set is carved from the train fold.

    Stratification is by each subject's majority stress label across all their
    audio tasks, ensuring each fold has a similar stressed/no-stress subject ratio.
    This prevents the model collapse seen with plain KFold on small datasets.

    Note: config.HELD_OUT_SUBJECTS are always excluded from the CV pool so
    they can never leak into training or validation.
    """
    n_folds = n_folds or config.NUM_FOLDS
    seed = seed or config.RANDOM_SEED
    val_ratio = val_ratio or config.VAL_RATIO

    all_subjects = get_all_audio_subjects()
    if not all_subjects:
        raise RuntimeError(
            f"No subjects found in {config.MEL_DIR}. "
            "Run feature extraction first: python feature_extraction/audio_processor.py --split train"
        )

    # Exclude held-out subjects from the CV pool
    held_out_set = set(config.HELD_OUT_SUBJECTS)
    subjects = np.array([s for s in all_subjects if s not in held_out_set])

    # Compute each subject's dominant binary-stress label for stratification
    labels_df = pd.read_csv(config.LABELS_CSV).set_index("subject/task")
    subject_labels = []
    for s in subjects:
        task_labels = []
        for task in config.AUDIO_TASKS:
            key = f"{s}_{task}"
            if key in labels_df.index:
                task_labels.append(int(labels_df.loc[key, "binary-stress"]))
        # Majority vote across tasks; default to 1 (stressed) if no data
        dominant = int(np.round(np.mean(task_labels))) if task_labels else 1
        subject_labels.append(dominant)
    subject_labels = np.array(subject_labels)

    # StratifiedKFold ensures each fold mirrors the overall class distribution
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    splits = list(skf.split(subjects, subject_labels))
    train_val_idx, test_idx = splits[fold]

    test_subjects = subjects[test_idx].tolist()
    train_val_subjects = subjects[train_val_idx]
    train_val_labels = subject_labels[train_val_idx]

    # Carve out validation from train (also stratified)
    np.random.seed(seed + fold)
    n_val = max(1, int(len(train_val_subjects) * val_ratio))
    # Stratified val carve: pick proportionally from each class
    stressed_idx = np.where(train_val_labels == 1)[0]
    no_stress_idx = np.where(train_val_labels == 0)[0]
    np.random.shuffle(stressed_idx)
    np.random.shuffle(no_stress_idx)
    # Proportion of val from each class
    n_val_stressed = max(1, round(n_val * len(stressed_idx) / len(train_val_subjects)))
    n_val_no_stress = max(0, n_val - n_val_stressed)
    val_idx = np.concatenate([stressed_idx[:n_val_stressed], no_stress_idx[:n_val_no_stress]])
    train_idx = np.concatenate([stressed_idx[n_val_stressed:], no_stress_idx[n_val_no_stress:]])
    val_subjects = train_val_subjects[val_idx].tolist()
    train_subjects = train_val_subjects[train_idx].tolist()

    return train_subjects, val_subjects, test_subjects
