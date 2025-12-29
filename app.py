import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image

MODEL_PATH = "coconut_disease_model.keras"

app = Flask(__name__)

# ===============================
# MANUAL CORS HEADERS (FIX)
# ===============================
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASS_NAMES = [
    "Bud Root",
    "Leaf Rot",
    "Gray Leaf Spot",
    "Stem Bleeding",
    "Bud Root Dropping"
]

def predict(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    idx = int(np.argmax(preds))

    return {
        "disease": CLASS_NAMES[idx],
        "confidence": round(float(preds[idx]) * 100, 2)
    }

@app.route("/", methods=["GET"])
def home():
    return "Coconut Disease Prediction API is running."

@app.route("/api/predict", methods=["POST", "OPTIONS"])
def generate():
    # Preflight request
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(silent=True)
    if not data or "imageSrc" not in data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        img_bytes = base64.b64decode(data["imageSrc"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    result = predict(img)
    return jsonify(result)
