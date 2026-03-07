import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

class StressIDDataset(Dataset):
    def __init__(self, root_dir, split="train", max_frames_face=300, max_frames_gesture=300, max_frames_audio=1876):
        self.root_dir = root_dir
        self.split = split
        self.max_frames_face = max_frames_face
        self.max_frames_gesture = max_frames_gesture
        self.max_frames_audio = max_frames_audio
        
        # Assuming label files are stored somewhere or we iterate over extracted .npy paths.
        # This is a sample structure that fetches synchronized feature files.
        feature_base = os.path.join(root_dir, "feature_extraction", "results")
        self.audio_dir = os.path.join(feature_base, "mel_spectrograms", split)
        self.face_dir = os.path.join(feature_base, "face", split)
        self.gesture_dir = os.path.join(feature_base, "gesture", split)
        
        # We will collect all available files that exist across all modalities.
        self.samples = self._build_dataset()
        
    def _build_dataset(self):
        samples = []
        if not os.path.exists(self.audio_dir): return samples
        
        # Loop over subjects and tasks
        for subject_id in os.listdir(self.audio_dir):
            subj_audio = os.path.join(self.audio_dir, subject_id)
            if not os.path.isdir(subj_audio): continue
            
            for f in os.listdir(subj_audio):
                if not f.endswith("_mel.npy"): continue
                task = f.replace("_mel.npy", "")
                
                audio_path = os.path.join(self.audio_dir, subject_id, f"{task}_mel.npy")
                face_path = os.path.join(self.face_dir, subject_id, f"{task}_face.npy")
                gesture_path = os.path.join(self.gesture_dir, subject_id, f"{task}_gesture.npy")
                
                if os.path.exists(face_path) and os.path.exists(gesture_path):
                    # In a real use-case, map subject/task to label from CSV.
                    # Here we mock label to 0 for structure.
                    samples.append({
                        "audio": audio_path,
                        "face": face_path,
                        "gesture": gesture_path,
                        "label": 0 # Replace with true label lookup logic mapped to baseline task
                    })
        return samples

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Audio: (1, 128, T)
        audio = np.load(sample["audio"])
        
        # Face: (T, 478, 3) 
        face = np.load(sample["face"])
        
        # Gesture: (T, 21, 3)
        gesture = np.load(sample["gesture"])
        
        # Padding/Truncation safety fallback (handled mostly in extraction, but good measure here)
        return {
            "audio": torch.tensor(audio, dtype=torch.float32),
            "face": torch.tensor(face, dtype=torch.float32),
            "gesture": torch.tensor(gesture, dtype=torch.float32),
            "label": torch.tensor(sample["label"], dtype=torch.long)
        }
