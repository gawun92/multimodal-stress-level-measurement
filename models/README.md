# 🧠 Deep Learning Architectures (Stage 2)

This directory contains the PyTorch network infrastructure of our Multimodal Pipeline. 

Our implementation completely overhauls the original baseline architecture to factor in **explicit sequences of time** rather than crushing data into flat CSV summaries. 

---

## 🏗 The Branches

### `face_branch.py` (Visual Modeling)
*   **Architecture:** 2-Layer LSTM (Long Short-Term Memory Network).
*   **Mechanic:** Receives sequences of dimension `(T, 478, 3)` from the `face_feature_extractor`. LSTMs are engineered explicitly for temporal sequences; this allows the model to map the gradual changing tension of facial muscles across extended video segments. 

### `gesture_branch.py` (Kinematic Modeling)
*   **Architecture:** 1D-CNN (Temporal Convolutional Network).
*   **Mechanic:** Receives flattened hand coordinate sequences of dimension `(T, 21, 3)`. We apply 1D temporal convolutions to scrub across the time axis, aggressively tracking isolated, momentary nervous ticks (such as sudden hand-wringing).

### `audio_branch.py` (Acoustic Modeling)
*   **Architecture:** 2D-CNN (Image-Like Object Classifier).
*   **Mechanic:** Receives `(1, 128, T)` Mel Spectrogram arrays. By running audio backwards as a 2D image, this branch operates as a highly trainable internal feature extractor, directly correlating human-audible frequency ranges with stress markers. 

---

## 🔀 Multimodal Fusion

### `fusion.py`
The original pipeline primarily relied on shallow Machine Learning algorithms or decision-level voting. We implement **Late Feature-Level Fusion**.

Our fusion logic uses an MLP that takes dense representations from our three modalities simultaneously:
1.  **Audio Embedding** (128-d)
2.  **Face Embedding** (128-d)
3.  **Gesture Embedding** (128-d)

These components concatenate into a single massive 384-dimensional tensor (`torch.cat`), passing into a **3-Layer Batch-Normalized MLP with a Dropout rate of 0.5**. This enables the model to map highly complex interactions (e.g., discovering whether high vocal tension combined with low facial tension equals baseline stress or anomalous excitement) while strictly avoiding overfitting.
