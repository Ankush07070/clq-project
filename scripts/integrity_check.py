"""Step 3: data integrity check for HAM10000 images vs metadata CSV."""

import os
import random
import sys
from collections import Counter

import pandas as pd
from PIL import Image

CSV_PATH = r"C:\Users\7CIN\Desktop\clq-project\data\raw\HAM10000_metadata.csv"
IMG_DIR = r"C:\Users\7CIN\Desktop\clq-project\data\raw\images"
JPEG_MAGIC = b"\xff\xd8"


def main():
    df = pd.read_csv(CSV_PATH)
    expected = {f"{x}.jpg" for x in df["image_id"].astype(str)}
    actual = {f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")}

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    print(f"CSV rows:            {len(df)}")
    print(f"Expected images:     {len(expected)}")
    print(f"JPGs found on disk:  {len(actual)}")
    print(f"Missing from disk:   {len(missing)}")
    print(f"Extra on disk:       {len(extra)}")
    if missing:
        print("  first missing:", missing[:5])
    if extra:
        print("  first extra:", extra[:5])

    # Zero-byte and JPEG magic-byte scan over all matched files
    zero_byte, bad_magic = [], []
    for name in sorted(expected & actual):
        p = os.path.join(IMG_DIR, name)
        if os.path.getsize(p) == 0:
            zero_byte.append(name)
            continue
        with open(p, "rb") as fh:
            if fh.read(2) != JPEG_MAGIC:
                bad_magic.append(name)
    print(f"Zero-byte files:     {len(zero_byte)}")
    print(f"Non-JPEG headers:    {len(bad_magic)}")

    # Full PIL decode check on a random sample of 300
    sample = random.Random(42).sample(sorted(expected & actual), 300)
    unreadable = []
    for name in sample:
        try:
            with Image.open(os.path.join(IMG_DIR, name)) as im:
                im.verify()
        except Exception as e:
            unreadable.append((name, repr(e)))
    print(f"PIL decode failures (of 300 sampled): {len(unreadable)}")
    for name, err in unreadable[:5]:
        print("  ", name, err)

    # Class distribution
    dist = Counter(df["dx"])
    full = {"nv": "Melanocytic nevi (benign)", "mel": "Melanoma",
            "bkl": "Benign keratosis-like", "bcc": "Basal cell carcinoma",
            "akiec": "Actinic keratoses", "vasc": "Vascular lesions",
            "df": "Dermatofibroma"}
    print("\nClass distribution:")
    for dx, n in dist.most_common():
        print(f"  {dx:6s} {n:5d}  ({n / len(df):5.1%})  {full.get(dx, '?')}")

    ok = not missing and not zero_byte and not bad_magic and not unreadable
    print(f"\nRESULT: {'PASS - ready for training' if ok else 'ISSUES FOUND - see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
