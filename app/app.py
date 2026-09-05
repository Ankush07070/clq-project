import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

from app.image_utils import process_image
from app.model_loader import load_model
from src.components.gradcam import GradCAM

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "artifacts" / "best_model.pth"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

CLASSES = [
    "Melanocytic Nevi",
    "Melanoma",
    "Benign Keratosis",
    "Basal Cell Carcinoma",
    "Actinic Keratoses",
    "Vascular Lesions",
    "Dermatofibroma",
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app, resources={r"/*": {"origins": os.getenv("FRONTEND_ORIGIN", "*")}})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
model_config = {}
model_error = None

try:
    model, model_config = load_model(str(MODEL_PATH), device)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as exc:
    model_error = str(exc)
    print(f"Model loading failed: {exc}")


def encode_heatmap(image, cam):
    original = np.array(image.convert("RGB").resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 0.60, heatmap, 0.40, 0)
    success, buffer = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("Could not encode Grad-CAM image")
    return base64.b64encode(buffer).decode("utf-8")


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"Image exceeds the {MAX_UPLOAD_MB} MB upload limit."}), 413


@app.get("/")
def home():
    return jsonify({"service": "Skin Lesion Classification API", "status": "running"})


@app.get("/health")
def health():
    return jsonify({
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "device": str(device),
        "model_path": str(MODEL_PATH),
        "model_error": model_error,
    }), (200 if model is not None else 503)


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model is not available on the server."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form field 'file'."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No filename supplied."}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Uploaded file is not a valid image."}), 400

    input_tensor = process_image(image).to(device)

    try:
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)

        class_idx = int(pred.item())
        response = {
            "prediction": CLASSES[class_idx],
            "class_index": class_idx,
            "confidence": float(confidence.item()),
            "probabilities": {
                CLASSES[i]: float(probs[0, i].item()) for i in range(len(CLASSES))
            },
        }

        # Grad-CAM requires gradients, so it is intentionally outside no_grad().
        try:
            target_layer = model.cnn[-1]
            gradcam = GradCAM(model, target_layer)
            cam = gradcam.generate(input_tensor, class_idx=class_idx, output_size=(224, 224))
            response["heatmap"] = encode_heatmap(image, cam)
            gradcam.close()
        except Exception as cam_error:
            response["heatmap"] = None
            response["explainability_warning"] = f"Grad-CAM unavailable: {cam_error}"

        return jsonify(response)
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Please try another image."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=False)
