import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

import config
from models.audio_branch import AudioBranch, AudioClassifier
from dataset import StressAudioDataset, get_all_audio_subjects

EMBED_DIR = os.path.join(os.path.dirname(__file__), "embeddings")
os.makedirs(EMBED_DIR, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_seed_models(fold, label, seeds=None):
    if seeds is None:
        seeds = config.ENSEMBLE_SEEDS
    num_classes = config.NUM_CLASSES_BINARY if label == "binary-stress" else config.NUM_CLASSES_AFFECT3

    models = []
    for seed in seeds:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"audio_branch_fold{fold}_seed{seed}_{label}.pt")
        if not os.path.exists(ckpt_path):
            print(f"checkpoint not found: {ckpt_path}, skipping")
            continue
        branch = AudioBranch(
            n_mels=config.N_MELS, max_frames=config.MAX_FRAMES,
            cnn_channels=config.CNN_CHANNELS, embed_dim=config.EMBED_DIM,
            n_heads=config.TRANSFORMER_HEADS, n_layers=config.TRANSFORMER_LAYERS,
            ff_dim=config.TRANSFORMER_FF_DIM, dropout=config.TRANSFORMER_DROPOUT,
        )
        model = AudioClassifier(branch, num_classes=num_classes)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE).eval()
        models.append(model)
        print(f"loaded seed {seed}")

    return models


def extract(fold=2, label="binary-stress"):
    print(f"extracting embeddings — fold {fold}, label {label}")

    models = load_seed_models(fold, label)
    if not models:
        raise RuntimeError(f"no checkpoints found for fold {fold}, run training first")

    all_subjects = get_all_audio_subjects()
    ds = StressAudioDataset(all_subjects, label_col=label, augment=False, windowed=False)
    print(f"{len(ds)} clips found")

    embeddings = {}
    labels_map = {}

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for i, (mel, lbl) in enumerate(tqdm(loader, desc="extracting")):
            mel = mel.to(DEVICE)
            seed_embs = [m.backbone.get_embedding(mel) for m in models]
            emb = torch.stack(seed_embs).mean(dim=0)

            npy_path, _ = ds.samples[i]
            parts = Path(npy_path).parts
            subject_id = parts[-2]
            task = Path(npy_path).stem.replace("_mel", "")
            key = f"{subject_id}/{task}"

            embeddings[key] = emb.squeeze(0).cpu().numpy()
            labels_map[key] = int(lbl.item())

    out_path = os.path.join(EMBED_DIR, f"audio_embeddings_fold{fold}_{label}.pt")
    payload = {
        "fold": fold,
        "label": label,
        "seeds": config.ENSEMBLE_SEEDS,
        "n_models": len(models),
        "embed_dim": 128,
        "n_clips": len(embeddings),
        "embeddings": {k: torch.tensor(v) for k, v in embeddings.items()},
        "labels": labels_map,
    }
    torch.save(payload, out_path)
    print(f"saved {len(embeddings)} embeddings to {out_path}")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--label", type=str, default="binary-stress",
                        choices=["binary-stress", "affect3-class"])
    args = parser.parse_args()
    extract(fold=args.fold, label=args.label)
