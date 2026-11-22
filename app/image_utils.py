from PIL import Image
from torchvision import transforms

from src.components.transforms import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

import numpy as np

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def process_image(image: Image.Image):
    """Inference preprocessing must exactly match validation preprocessing."""
    return INFERENCE_TRANSFORM(image).unsqueeze(0)


def skin_fraction(image: Image.Image, size: int = 112) -> float:
    """Fraction of pixels matching a standard RGB skin-tone rule.

    Real HAM10000 lesion photos are dominated by skin pixels (mean fraction
    ~0.77); grey paper, flat fills and most non-skin junk score near zero.
    Used by the OOD gate as a cheap physical pre-check.
    """
    small = image.convert("RGB").resize((size, size))
    a = np.asarray(small, dtype=np.float32)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    rule = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & ((r - g) > 15) & ((r - b) > 15)
    return float(rule.mean())
