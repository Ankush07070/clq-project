"""Build the KNN out-of-distribution (OOD) gate.

Extracts 256-d pooled embeddings from the trained model for every training
image (the "feature bank"), then calibrates a distance threshold on the
validation split (disjoint lesions, so no self-matches). At inference, an
input whose embedding is far from its nearest training neighbours is
rejected as not-a-skin-lesion (KNN-OOD, Sun et al. 2022).

Run from the project root:  python scripts/build_feature_bank.py

Creates:
    artifacts/feature_bank.npz   bank embeddings + labels
    artifacts/ood_config.json    k, threshold, calibration stats
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model_loader import load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "artifacts")
K = 5
BATCH_SIZE = 64


def file_sha256(path, buf_size=1 << 20):
    """Fingerprint the checkpoint so the gate can refuse cross-model banks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def extract_split(model, device, df):
    """L2-normalized embeddings for every image in the dataframe."""
    from src.components.transforms import get_val_transforms

    transform = get_val_transforms()
    chunks, batch = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="embeddings"):
        img = Image.open(row["image_path"]).convert("RGB")
        batch.append(transform(img))
        if len(batch) == BATCH_SIZE:
            x = torch.stack(batch).to(device)
            chunks.append(model.extract_features(x).cpu())
            batch = []
    if batch:
        x = torch.stack(batch).to(device)
        chunks.append(model.extract_features(x).cpu())
    feats = torch.cat(chunks)
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)
    return feats.numpy().astype(np.float32)


def knn_distances(query, bank, k=K):
    """Mean distance to the k nearest bank vectors for each query row."""
    sims = query @ bank.T  # cosine similarity (both sides L2-normalized)
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    top_sims = np.take_along_axis(sims, idx, axis=1)
    dists = np.sqrt(np.clip(2.0 - 2.0 * top_sims, 0.0, None))
    return dists.mean(axis=1)


def main():
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(ARTIFACTS, "best_model.pth")
    model, _ = load_model(ckpt_path, device)
    ckpt_sha = file_sha256(ckpt_path)

    train_df = pd.read_csv(os.path.join(ROOT, "data", "processed", "train.csv"))
    val_df = pd.read_csv(os.path.join(ROOT, "data", "processed", "val.csv"))
    print(f"train images: {len(train_df)}, val images: {len(val_df)}")

    bank = extract_split(model, device, train_df)
    val = extract_split(model, device, val_df)

    val_dist = knn_distances(val, bank)
    p95, p99 = np.percentile(val_dist, [95, 99])
    threshold = float(p99)

    np.savez_compressed(
        os.path.join(ARTIFACTS, "feature_bank.npz"),
        features=bank,
        labels=train_df["label"].values,
        checkpoint_sha256=np.array(ckpt_sha),
    )
    config = {
        "k": K,
        "metric": "mean cosine distance to k nearest, L2-normalized 256-d embeddings",
        "bank_size": int(bank.shape[0]),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_backbone": getattr(model, "backbone_name", "unknown"),
        "threshold": threshold,
        "val_p95": float(p95),
        "val_p99": float(p99),
        "val_mean": float(val_dist.mean()),
        "val_max": float(val_dist.max()),
        "confidence_floor": 0.25,
    }
    with open(os.path.join(ARTIFACTS, "ood_config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    print(f"bank: {bank.shape[0]} x {bank.shape[1]}")
    print(f"val distances  mean={config['val_mean']:.4f}  p95={p95:.4f}  "
          f"p99={p99:.4f}  max={config['val_max']:.4f}")
    print(f"OOD threshold set to: {threshold:.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
