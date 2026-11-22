"""KNN out-of-distribution gate over the trained model's embeddings.

An input embedding is compared against the bank of training embeddings;
if its mean distance to the k nearest neighbours exceeds the calibrated
threshold (or softmax confidence is suspiciously low) the request is
rejected as "not a skin lesion" instead of forcing a 7-class diagnosis.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
BANK_PATH = Path(os.getenv("FEATURE_BANK_PATH", BASE_DIR / "artifacts" / "feature_bank.npz"))
CONFIG_PATH = Path(os.getenv("OOD_CONFIG_PATH", BASE_DIR / "artifacts" / "ood_config.json"))


class OODGate:
    def __init__(self, device):
        self.device = device
        self.bank = None
        self.config = {}
        self.enabled = BANK_PATH.exists() and CONFIG_PATH.exists()
        if not self.enabled:
            print(f"OOD gate disabled (missing {BANK_PATH.name} or {CONFIG_PATH.name})")
            return
        data = np.load(BANK_PATH)
        self.bank = torch.from_numpy(data["features"]).to(device)
        with open(CONFIG_PATH) as fh:
            self.config = json.load(fh)
        self.k = int(os.getenv("OOD_K", self.config.get("k", 5)))
        self.threshold = float(os.getenv("OOD_THRESHOLD", self.config.get("threshold", 0.6)))
        self.confidence_floor = float(
            os.getenv("CONFIDENCE_FLOOR", self.config.get("confidence_floor", 0.25))
        )
        print(f"OOD gate armed: bank={self.bank.shape[0]} vectors, "
              f"k={self.k}, threshold={self.threshold:.4f}")

    @torch.no_grad()
    def distance(self, features):
        """Mean cosine distance to the k nearest bank embeddings."""
        query = torch.nn.functional.normalize(features, p=2, dim=1)
        sims = query @ self.bank.T  # (1, N)
        k = min(self.k, self.bank.shape[0])
        top = torch.topk(sims, k, dim=1).values
        dists = torch.sqrt(torch.clamp(2.0 - 2.0 * top, min=0.0))
        return float(dists.mean().item())

    def check(self, features, confidence, skin_frac=None):
        """Return a verdict dict: rejected, reason, distance, score, threshold."""
        if not self.enabled:
            return {"rejected": False, "reason": None, "distance": None,
                    "score": None, "threshold": None, "skin_fraction": skin_frac}
        dist = self.distance(features)
        floor = float(os.getenv(
            "SKIN_FRACTION_FLOOR", self.config.get("skin_fraction_floor", 0.05)
        ))
        if "score_threshold" in self.config:
            # Layer 1 (physical): real lesion photos are skin-dominated.
            if skin_frac is not None and skin_frac < floor:
                return {"rejected": True, "reason": "no_skin_tone", "distance": dist,
                        "score": None, "threshold": floor, "skin_fraction": skin_frac}
            # Layer 2 (learned): junk has high distance AND low confidence.
            alpha = float(os.getenv("OOD_ALPHA", self.config.get("alpha", 1.0)))
            threshold = float(os.getenv("OOD_SCORE_THRESHOLD", self.config["score_threshold"]))
            score = dist + alpha * (1.0 - confidence)
            verdict = {"rejected": False, "reason": None, "distance": dist,
                       "score": score, "threshold": threshold, "skin_fraction": skin_frac}
            if score > threshold:
                verdict.update(rejected=True, reason="out_of_distribution_score")
            elif confidence < self.confidence_floor:
                verdict.update(rejected=True, reason="low_confidence")
            return verdict
        # Legacy distance-only calibration (feature bank without score calibration).
        verdict = {"rejected": False, "reason": None, "distance": dist,
                   "score": None, "threshold": self.threshold, "skin_fraction": skin_frac}
        if skin_frac is not None and skin_frac < floor:
            verdict.update(rejected=True, reason="no_skin_tone")
        elif dist > self.threshold:
            verdict.update(rejected=True, reason="distance_above_threshold")
        elif confidence < self.confidence_floor:
            verdict.update(rejected=True, reason="low_confidence")
        return verdict
