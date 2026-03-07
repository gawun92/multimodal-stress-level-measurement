# StressID Multimodal Pipeline (Group 14)

Welcome to the **Group 14** implementation of the StressID Multimodal Pipeline. This repository is specifically tailored for execution on the **USC CARC (Discovery/Antigravity cluster)**, leveraging custom algorithmic optimizations to overcome environmental I/O bottlenecks.

### Environment & Pathing Constraints

*   **Cluster:** USC CARC.
*   **CARC Project Root:** `/project2/msoleyma_1026/group_14/`
*   **Local Project Root:** Will dynamically fallback to `data/stressid/` if the CARC host is not detected.

### The "No-JPG" Architecture

To prevent choking the CARC compute nodes when dealing with high-frequency image data, we enforce a strict **No-JPG Rule**:
*   No `.jpg` or `.png` intermediate frames from the `StressID Dataset` videos are ever saved to disk.
*   Scripts use `cv2.VideoCapture` to read `.mp4` pixel blocks straight into RAM, buffer them at 5 FPS, and pass them to inference models inside memory boundaries.
*   All features (MediaPipe FaceMesh, Hands, Audio Mel Spectrograms) are compiled dynamically and saved out as chunked `.npy` tracking matrices.

### Directory Outline

```text
group_14/
├── data/stressid/              # Locally mapped raw data (ignored on CARC)
├── feature_extraction/         # Stage 1: Dynamic extractors (Raw -> .npy)
│   ├── audio_processor.py      # Output: 128-bin Mel Spectrograms
│   ├── face_extractor.py       # Output: 478 MediaPipe 3D Landmarks
│   └── gesture_extractor.py    # Output: 21 MediaPipe 3D Landmarks
├── models/                     # Stage 2: Deep Learning Neural Branches
│   ├── audio_branch.py         # 2D-CNN Sequence Architecture
│   ├── face_branch.py          # LSTM Sequence Architecture
│   ├── gesture_branch.py       # 1D-CNN Sequence Architecture
│   └── fusion.py               # Feature-Level Late Fusion (MLP)
├── dataset.py                  # Custom multimodal PyTorch Dataset loader
├── train.py                    # Multi-input training sequence w/ Loss Tracking
└── extract_features.slurm      # Master shell script for SLURM batch deployments
```

### Execution

To prep all features at once on the cluster, submit the provided batch script:

```bash
sbatch extract_features.slurm
```
