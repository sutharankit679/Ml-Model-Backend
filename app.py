import os

# ===============================
# 🔥 IMPORTANT: FORCE CPU ONLY
# ===============================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
IMAGE_SIZE = (224, 224)

app = Flask(__name__)

# ===============================
# 🔥 LOAD MODEL ONCE (VERY IMPORTANT)
# ===============================
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

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
def predict_image(img: Image.Image):
    # Resize (HARD LIMIT for memory safety)
    img = img.resize(IMAGE_SIZE)

    # Normalize
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    # Predict
    preds = model.predict(img_arr, verbose=0)[0]
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
    return "Coconut Disease Prediction API is running."

# ===============================
# ML PREDICTION API
# ===============================
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "imageSrc" not in data:
        return jsonify({"error": "imageSrc missing"}), 400

    try:
        # Decode base64
        image_bytes = base64.b64decode(data["imageSrc"])
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({
            "error": "Invalid image data",
            "details": str(e)
        }), 400

    try:
        result = predict_image(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500
