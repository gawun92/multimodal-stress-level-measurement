# Agent Handoff Document: StressID Multimodal Pipeline (Group 14)
**Last Updated**: March 10, 2026

---

## Project Overview

CSCI 535 Group 14 — multimodal stress detection on the **StressID dataset** (NeurIPS 2023).
Three-branch pipeline (Audio / Face / Gesture) with late fusion.
Target cluster: USC CARC Discovery/Antigravity at `/project2/msoleyma_1026/group_14/`.

---

## Audio Branch — Current State (COMPLETE)

### Architecture
```
Input: (B, 1, 128, 1876)  mel spectrogram  [16kHz, 128-bin, ~60s audio]
  ConvBlock 1:  Conv2d(1->32)   + BN + ReLU + MaxPool(2,2) -> (B, 32, 64, 938)
  ConvBlock 2:  Conv2d(32->64)  + BN + ReLU + MaxPool(2,2) -> (B, 64, 32, 469)
  ConvBlock 3:  Conv2d(64->128) + BN + ReLU + MaxPool(2,2) -> (B, 128, 16, 234)
  Freq-axis mean pool                                       -> (B, 128, 234)
  Permute -> (B, 234, 128)  [time-first]
  Sinusoidal Positional Encoding
  TransformerEncoder (2 layers, 4 heads, ff_dim=256, dropout=0.1)
  AttentionPooling (learnable query vector)                 -> (B, 128)
  MLP Head: Linear(128->64) + ReLU + Dropout(0.3) + Linear(64->num_classes)
  Output: raw logits (B, 2) — no sigmoid, CrossEntropyLoss applied externally
Total parameters: ~366,594
```

**Key design choices:**
- No sigmoid output: two competing logits, `argmax` decides prediction. `softmax` only applied
  externally for probability outputs (AUC-ROC). Generalizes to 3-class affect labels with no
  code changes (just `num_classes=3`).
- Class-weighted CrossEntropyLoss (`compute_class_weight("balanced")`) for 71%/29% imbalance
- SpecAugment: FrequencyMasking(15) + TimeMasking(35), train-only, toggled via `--augment` flag
- Attention Pooling (not mean pooling) — learnable weighted aggregation over 234 time steps,
  per professor suggestion to preserve temporal axis
- Sinusoidal PE (Vaswani et al.) for position awareness in the Transformer

### Hyperparameters (config.py)

| Param | Value | Note |
|---|---|---|
| LEARNING_RATE | 5e-5 | Tuned down from 1e-4 to reduce oscillation |
| WEIGHT_DECAY | 1e-3 | L2 regularization, tuned up from 1e-4 |
| BATCH_SIZE | 16 | |
| NUM_EPOCHS | 50 | with early stopping (PATIENCE=10) |
| NUM_FOLDS | 5 | StratifiedKFold on subject-level majority label |
| VAL_RATIO | 0.15 | Stratified val carve (proportional class split) |
| RANDOM_SEED | 42 | |
| DEVICE | cuda -> mps -> cpu | Auto-detected (MPS = Apple Silicon) |

### Held-Out Subjects (team convention)
```python
HELD_OUT_SUBJECTS = ["wssm", "x1q3", "y8c3", "y9z6"]
```
These 4 subjects are **never used in training or CV**. They are only evaluated once after CV
is complete. Best CV fold is selected by Macro F1 (not accuracy) to avoid picking collapsed folds.

---

## Latest CV Results (StratifiedKFold, 50 subjects, binary-stress)

```
Fold 0: acc=0.600 | F1w=0.547 | F1m=0.433 | AUC=0.504 | BalAcc=0.462 | MCC=-0.101
Fold 1: acc=0.786 | F1w=0.691 | F1m=0.440 | AUC=0.229 | BalAcc=0.500 | MCC= 0.000  <- COLLAPSED
Fold 2: acc=0.657 | F1w=0.667 | F1m=0.643 | AUC=0.707 | BalAcc=0.667 | MCC= 0.314  <- BEST
Fold 3: acc=0.671 | F1w=0.680 | F1m=0.578 | AUC=0.603 | BalAcc=0.583 | MCC= 0.158
Fold 4: acc=0.671 | F1w=0.594 | F1m=0.440 | AUC=0.580 | BalAcc=0.485 | MCC=-0.053

Mean Accuracy:     0.677 +/- 0.060
Mean Weighted F1:  0.636 +/- 0.056
Mean Macro F1:     0.507 +/- 0.087   <- first time >0.50 (model learns minority class signal)
Mean AUC-ROC:      0.524 +/- 0.161   <- above 0.50 (real separability)
Mean Balanced Acc: 0.539 +/- 0.076
Mean MCC:          0.064 +/- 0.153

Held-out (4 subjects, used Fold 1 ckpt -- was collapsed, next run will use Fold 2):
  Acc=0.536 | F1m=0.349 | AUC=0.456 | MCC=0.000
```

**Fold 1 collapse explained:** acc=0.786 but MCC=0.000 and AUC=0.229 — the model predicted
everything as "stressed". Since 79% of test samples ARE stressed, this gives a spuriously high
accuracy. The `best_fold` selector is now fixed to use `argmax(macro_f1)` (-> Fold 2) instead
of `argmax(accuracy)`. The held-out results will be correct on the next full training run.

---

## Metrics Used (Why NOT R-squared)

R-squared is a **regression** metric that measures explained variance. It is meaningless for
binary classification — never use it here.

| Metric | Range | Random | What it catches |
|---|---|---|---|
| Accuracy | 0-1 | ~71% | misleading for imbalanced classes |
| Weighted F1 | 0-1 | ~60% | accounts for support, still majority-biased |
| **Macro F1** | 0-1 | ~50% | equal weight both classes — primary metric |
| **AUC-ROC** | 0-1 | 0.50 | threshold-independent separability |
| **Balanced Acc** | 0-1 | 0.50 | (sensitivity + specificity) / 2, collapse -> 0.5 |
| **MCC** | -1 to +1 | 0 | gold standard for imbalanced binary, collapse -> 0 |

Best fold selection: `argmax(macro_f1)` — most robust against fold collapse.

---

## Files (Audio Branch)

| File | Status | Description |
|---|---|---|
| `config.py` | DONE | All hyperparameters and paths (CARC/local auto-detect) |
| `dataset.py` | DONE | StratifiedKFold splits, held-out exclusion, StressAudioDataset |
| `models/audio_branch.py` | DONE | CNN + Transformer + AttentionPooling + AudioClassifier |
| `train.py` | DONE | Single-fold trainer (`--fold`, `--label`, `--augment`) |
| `evaluate.py` | DONE | Single-fold eval with all 6 metrics |
| `run_all_folds.py` | DONE | 5-fold CV orchestrator + held-out eval, saves cv_results.json |
| `visualize_cv.py` | DONE | 7 figures from cv_results.json including professor_dashboard.png |
| `feature_extraction/audio_processor.py` | DONE | Mel spectrogram extractor (no-JPG rule) |
| `cv_results.json` | DATA | Latest 5-fold results (preds/labels/probs stored per fold) |
| `figures/` | DATA | All generated visualizations |

### Files NOT in git (too large / generated at runtime)
- `feature_extraction/results/` — 347MB of extracted .npy mel spectrograms
- `checkpoints/` — .pt model weights (~10MB)
- `*.log` / `training_log.txt` — runtime logs (regenerated each run)

---

## How to Run

```bash
# 1. Extract mel spectrograms from raw .wav files
python feature_extraction/audio_processor.py

# 2. Run full 5-fold CV (trains + evaluates all folds, saves cv_results.json + cv_results.log)
python run_all_folds.py

# 3. Regenerate all 7 figures from saved results
python visualize_cv.py

# 4. Train a single fold manually
python train.py --fold 2 --label binary-stress --augment

# 5. Evaluate a single fold from its checkpoint
python evaluate.py --fold 2 --label binary-stress
```

For SLURM on CARC:
```bash
sbatch extract_features.slurm   # 8 CPUs, 32GB, 12h
```

---

## Professor Constraints / Design Decisions

- **No-JPG rule**: Video frames are never saved to disk, streamed into RAM via `cv2.VideoCapture`
- **Subject-level CV**: Every split is by *subject*, never by sample — no data leakage
- **StratifiedKFold**: Stratify on each subject's dominant label to prevent fold collapse
- **Attention Pooling**: Replaces mean pooling per professor suggestion
- **Two logits, not sigmoid**: `CrossEntropyLoss(num_classes)` generalizes to 3-class affect

---

## Next Steps

1. **Re-run audio CV**: `python run_all_folds.py` — held-out will now use Fold 2 (non-collapsed)
2. **Face branch** (`models/face_branch.py`): 2-layer LSTM on (T, 478, 3) MediaPipe landmarks
3. **Gesture branch** (`models/gesture_branch.py`): 1D-CNN on (T, 21, 3) hand landmarks
4. **Fusion** (`fusion.py`): 3-layer BN-MLP on 384-d concat of all three 128-d embeddings
5. **dataset.py extension**: Add face/gesture modalities alongside audio

---

*Always read this document first when picking up this project.*
*Update the "Latest CV Results" section after each significant training run.*
