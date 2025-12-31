import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "coconut_disease_model.keras"

app = Flask(__name__)

# ===============================
# LOAD MODEL (ONCE)
# ===============================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASS_NAMES = [
    "Bud Root",
    "Leaf Rot",
    "Gray Leaf Spot",
    "Stem Bleeding",
    "Bud Root Dropping"
]

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict(img: Image.Image):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    idx = int(np.argmax(preds))

    return {
        "disease": CLASS_NAMES[idx],
        "confidence": round(float(preds[idx]) * 100, 2)
    }

# ===============================
# HEALTH CHECK
# ===============================
@app.route("/", methods=["GET"])
def home():
    return "ML Backend is running."

# ===============================
# ML PREDICTION API
# (CALLED ONLY BY VERCEL SERVER)
# ===============================
@app.route("/api/predict", methods=["POST"])
def generate():
    data = request.get_json(silent=True)

    if not data or "imageSrc" not in data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        img_bytes = base64.b64decode(data["imageSrc"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    result = predict(img)
    return jsonify(result)
