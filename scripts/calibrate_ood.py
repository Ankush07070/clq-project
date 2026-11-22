"""Calibrate the combined OOD score on the validation split.

Real lesions tend to have LOW nearest-neighbour distance AND HIGH softmax
confidence; junk inputs score badly on both. The gate therefore uses
    combined score = distance + alpha * (1 - confidence)
with the threshold taken from the validation score distribution.

Run from the project root:  python scripts/calibrate_ood.py [--alpha 1.0] [--percentile 95]

Updates artifacts/ood_config.json with alpha / score_threshold.
"""

import argparse
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
from src.components.transforms import get_val_transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "artifacts")
BANK_PATH = os.path.join(ARTIFACTS, "feature_bank.npz")
CONFIG_PATH = os.path.join(ARTIFACTS, "ood_config.json")
BATCH_SIZE = 64


@torch.no_grad()
def extract_with_conf(model, device, df):
    """(L2-normalized embeddings, softmax confidences) for each image."""
    transform = get_val_transforms()
    feats, confs, batch = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="val embeddings"):
        img = Image.open(row["image_path"]).convert("RGB")
        batch.append(transform(img))
        if len(batch) == BATCH_SIZE:
            x = torch.stack(batch).to(device)
            f = model.extract_features(x)
            confs.append(torch.softmax(model.classifier(f), dim=1).max(dim=1).values.cpu())
            feats.append(torch.nn.functional.normalize(f, p=2, dim=1).cpu())
            batch = []
    if batch:
        x = torch.stack(batch).to(device)
        f = model.extract_features(x)
        confs.append(torch.softmax(model.classifier(f), dim=1).max(dim=1).values.cpu())
        feats.append(torch.nn.functional.normalize(f, p=2, dim=1).cpu())
    return torch.cat(feats).numpy(), torch.cat(confs).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--percentile", type=float, default=95.0)
    args = parser.parse_args()

    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(os.path.join(ARTIFACTS, "best_model.pth"), device)

    bank = np.load(BANK_PATH)["features"]
    bank_t = torch.from_numpy(bank).to(device)

    val_df = pd.read_csv(os.path.join(ROOT, "data", "processed", "val.csv"))
    val_feats, val_conf = extract_with_conf(model, device, val_df)

    sims = val_feats @ bank.T
    idx = np.argpartition(-sims, 4, axis=1)[:, :5]
    top = np.take_along_axis(sims, idx, axis=1)
    dist = np.sqrt(np.clip(2.0 - 2.0 * top, 0.0, None)).mean(axis=1)

    print(f"val: dist mean={dist.mean():.3f} p95={np.percentile(dist, 95):.3f} | "
          f"conf mean={val_conf.mean():.3f} p5={np.percentile(val_conf, 5):.3f}")

    with open(CONFIG_PATH) as fh:
        config = json.load(fh)

    alpha = args.alpha
    pct = args.percentile
    score = dist + alpha * (1.0 - val_conf)
    threshold = float(np.percentile(score, pct))
    print(f"combined score (alpha={alpha}): mean={score.mean():.3f} "
          f"p90={np.percentile(score, 90):.3f} p95={np.percentile(score, 95):.3f} "
          f"p99={np.percentile(score, 99):.3f} max={score.max():.3f}")
    print(f"score threshold set to ({pct}th pct): {threshold:.3f}")

    # Skin-tone pre-check floor: keep it under the weakest 1% of real lesions.
    from app.image_utils import skin_fraction
    skin = np.array([skin_fraction(Image.open(p)) for p in val_df["image_path"]])
    val_p1 = float(np.percentile(skin, 1))
    skin_floor = round(min(0.05, val_p1), 3)
    print(f"val skin-fraction: p1={val_p1:.3f} p5={np.percentile(skin, 5):.3f} "
          f"mean={skin.mean():.3f} -> floor={skin_floor}")

    config["alpha"] = alpha
    config["score_threshold"] = threshold
    config["score_percentile"] = pct
    config["val_score_mean"] = float(score.mean())
    config["skin_fraction_floor"] = skin_floor
    config["val_skin_fraction_p1"] = val_p1
    with open(CONFIG_PATH, "w") as fh:
        json.dump(config, fh, indent=2)

    # Report hard negatives (data/negatives/) under the final configuration.
    neg_dir = os.path.join(ROOT, "data", "negatives")
    if os.path.isdir(neg_dir):
        from app.model_loader import load_model as _lm  # already loaded above
        for name in sorted(os.listdir(neg_dir)):
            path = os.path.join(neg_dir, name)
            try:
                img = Image.open(path).convert("RGB")
            except OSError:
                continue
            from src.components.transforms import get_val_transforms
            x = get_val_transforms()(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model.extract_features(x)
                conf = float(torch.softmax(model.classifier(f), dim=1).max().item())
            verdict = ood_check_stub(f, conf, skin_fraction(img), config, bank, device)
            status = "REJECTED" if verdict["rejected"] else "accepted"
            print(f"negative {name:28s} -> {status} ({verdict['reason']}, "
                  f"score={verdict['score']}, skin={verdict['skin_fraction']:.3f})")
    print("DONE - ood_config.json updated")


def ood_check_stub(features, confidence, skin_frac, config, bank, device):
    """Mirror of app.ood_gate.OODGate.check for offline calibration reports."""
    import torch as _t
    fn = _t.nn.functional.normalize(features, p=2, dim=1)
    sims = fn @ _t.from_numpy(bank).to(device).T
    top = _t.topk(sims, 5, dim=1).values
    dist = float(_t.sqrt(_t.clamp(2.0 - 2.0 * top, min=0.0)).mean().item())
    if skin_frac < config.get("skin_fraction_floor", 0.05):
        return {"rejected": True, "reason": "no_skin_tone", "score": None,
                "skin_fraction": skin_frac}
    score = dist + config.get("alpha", 1.0) * (1.0 - confidence)
    if score > config.get("score_threshold", 9e9):
        return {"rejected": True, "reason": "out_of_distribution_score", "score": round(score, 3),
                "skin_fraction": skin_frac}
    if confidence < config.get("confidence_floor", 0.25):
        return {"rejected": True, "reason": "low_confidence", "score": round(score, 3),
                "skin_fraction": skin_frac}
    return {"rejected": False, "reason": None, "score": round(score, 3),
            "skin_fraction": skin_frac}


if __name__ == "__main__":
    main()
