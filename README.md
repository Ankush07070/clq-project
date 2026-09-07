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
```

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
