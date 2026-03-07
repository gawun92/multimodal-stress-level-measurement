# 🔬 Feature Extraction (Stage 1)

This directory contains the scripts responsible for preparing the raw `/data/stressid` inputs for deep learning consumption. 

### Core Constraint: The "No-JPG" Rule
Because traditional visual extraction requires rendering thousands of image files per minute, standard preprocessing cripples the USC CARC IO infrastructure. 

These scripts use **Direct Memory Access (DMA) Simulation**:
1. Initializing an in-memory `VideoCapture` buffer of the raw `.mp4`
2. Skipping frames on-the-fly to a target chunk weight (5 FPS)
3. Executing inference on the active RAM blocks
4. Concatenating final states directly into numpy `.npy` saves.

---

## The Extractors

### `face_feature_extractor.py`
*   **Input:** Raw `.mp4` tracking files.
*   **Engine:** `mediapipe` FaceMesh Landmarker Task.
*   **Mechanic:** Determines exactly 478 point-clouds for the face. Features robust failure handling; if a given frame contains no valid face mapping (e.g., subject turned away), it actively zero-pads the time frame to ensure consistent tensor boundaries.
*   **Output:** `(MaxFrames, 478, 3)` `.npy` tensor.

### `gesture_feature_extractor.py`
*   **Input:** Raw `.mp4` tracking files.
*   **Engine:** `mediapipe` Hand Landmarker Task.
*   **Mechanic:** Identifies 21 distinct kinematic joints across the visual subject's hand. Used for analyzing high-anxiety hand-wringing or rigidity.
*   **Output:** `(MaxFrames, 21, 3)` `.npy` tensor.

### `audio_processor.py`
*   **Input:** Raw `.wav` 
*   **Engine:** `librosa`
*   **Mechanic:** Immediately standardizes sampling frequencies to `16000Hz`. Pushes waveforms through a `1024` FFT window structure to yield energy buckets across traditional acoustic frequencies.
*   **Output:** `(1, 128, MaxFrames)` dimensional 128-Bin Mel Spectrograms (`.npy`).
