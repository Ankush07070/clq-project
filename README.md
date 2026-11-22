# Dermalens | Explainable Skin Lesion Classification

A full-stack computer-vision project for **7-class skin lesion classification** using a hybrid **MobileNetV2 + Transformer** model, PyTorch, Flask, React and Grad-CAM.

> Research/educational project only. It is not a medical diagnostic device.

## Pipeline

```text
HAM10000 metadata + images
        ↓
Path + image validation
        ↓
Group split by lesion_id
(train / validation / test)
        ↓
Training augmentation + WeightedRandomSampler
        ↓
MobileNetV2 CNN
        ↓
1×1 projection: 1280 → 256
        ↓
Multi-head self-attention Transformer
        ↓
Global pooling + classifier
        ↓
Best checkpoint by validation Macro-F1
        ↓
Clean test-set evaluation
        ↓
Flask /predict API
        ↓
Prediction + probabilities + Grad-CAM
        ↓
React frontend
```

## Classes

`nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`

## Important fixes in this version

- Split **before** any balancing/sampling to prevent leakage.
- Group split by `lesion_id` so the same lesion cannot cross train/validation/test.
- Use `WeightedRandomSampler` on the **training loader only**.
- Keep validation/test distributions untouched.
- Use the same `224×224 + ImageNet normalization` preprocessing for training validation and inference.
- Select the best model using validation **Macro-F1**, not accuracy alone.
- Generate a separate test-set report and confusion matrix.
- Replace deprecated Grad-CAM backward hooks with `register_full_backward_hook`.
- Return Grad-CAM as base64 from the API for the React UI.
- Add `/health`, upload limits and clearer API errors.
- Save model configuration with the checkpoint.
- Use a single frontend.
- Reject non-skin inputs with a **KNN out-of-distribution gate** (see below).

## Out-of-distribution gate (KNN-OOD)

A closed-set classifier always picks one of its 7 classes — even for a photo
of a notebook. To reject such inputs, the API embeds each image with the
trained model (256-d pooled features) and compares it against a bank of
training-set embeddings (`artifacts/feature_bank.npz`):

- `distance` = mean cosine distance to the 5 nearest training embeddings.
- `score` = `distance + alpha * (1 - confidence)` (junk scores badly on both).
- Thresholds calibrated on the validation split: distance = p99, score = p95
  (`artifacts/ood_config.json`).

If the score exceeds the threshold (or confidence < 0.25) the API returns
`"prediction": "Not a skin lesion"` instead of forcing a diagnosis.

Three rejection layers, calibrated on validation:

1. **Skin-tone check** (physical): fraction of pixels passing an RGB
   skin rule; real lesion photos average 0.77, junk often ~0. Floor at the
   1st percentile of validation skin fractions — catches grey paper,
   flat fills, handwriting photos.
2. **Combined KNN score** (learned): `distance + 1 x (1 - confidence)` at
   the 95th percentile — catches warm-toned junk that passes the skin
   check (textured cardboard, noise). By design ~5% of genuine lesion
   photos score above it and must retry.
3. **Confidence floor** 0.25.

Rebuild after retraining:

```bash
python scripts/build_feature_bank.py
python scripts/calibrate_ood.py
```

## Run the trained model on another machine

The complete matched artifact set (checkpoint, feature bank, OOD config,
metrics) is committed under `artifacts/v3_backup/`. A fresh clone needs no
retraining and no dataset — the API automatically falls back to that folder
when the root artifacts are absent:

```bash
git clone https://github.com/Ankush07070/clq-project.git
cd clq-project
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m app.app
```

On a machine without an NVIDIA GPU, install the plain (CPU) `torch` and
`torchvision` wheels instead — inference works, just slower.

Optional UI:

```bash
cd frontend && npm install && npm run dev
```

The feature bank stores the sha256 of the checkpoint it was built from;
`app/ood_gate.py` refuses to arm on a mismatch, so a stale bank can never
silently reject every image again.

## Training

Place the HAM10000 images under:

```text
data/raw/images/
```


and metadata at:

```text
data/raw/HAM10000_metadata.csv
```

Then:

```bash
pip install -r requirements.txt
python main.py
```

Optional environment variables:

```text
EPOCHS=15
BATCH_SIZE=32
LEARNING_RATE=0.0001
EARLY_STOPPING_PATIENCE=4
BACKBONE=efficientnet_b0      # mobilenet_v2 (default) | efficientnet_b0
LR_SCHEDULER=cosine           # plateau (default) | cosine
```

The final test evaluation uses test-time augmentation (4-flip averaging).

Training creates:

```text
artifacts/best_model.pth
artifacts/model_config.json
artifacts/training_history.json
artifacts/test_metrics.json

data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

## API

Run:

```bash
python -m app.app
```

or:

```bash
gunicorn --workers 1 --timeout 120 --bind 0.0.0.0:$PORT app.app:app
```

Endpoints:

```text
GET  /
GET  /health
POST /predict
```

Prediction request:

```text
multipart/form-data
file=<image>
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI shows the predicted class, confidence, all class probabilities and Grad-CAM attention visualization.
