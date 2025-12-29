import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "coconut_disease_model.keras"

# ===============================
# APP INIT
# ===============================
app = Flask(__name__)

# 🔥 SIMPLE GLOBAL CORS (FIXES FRONTEND ERROR)
# Allows requests from Vercel, localhost, etc.
CORS(app, supports_credentials=True)

# ===============================
# LOAD MODEL
# ===============================
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    raise e

CLASS_NAMES = [
    "Bud Root",
    "Leaf Rot",
    "Gray Leaf Spot",
    "Stem Bleeding",
    "Bud Root Dropping"
]

# ===============================
# UTILS
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
# ROUTES
# ===============================
@app.route("/", methods=["GET"])
def home():
    return "Coconut Disease Prediction API is running."

@app.route("/api/predict", methods=["POST"])
def generate():
    data = request.get_json(silent=True)

    if not data or "imageSrc" not in data:
        return jsonify({"error": "No image data provided"}), 400

    img_data = data.get("imageSrc")

    try:
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {str(e)}"}), 400

    result = predict(img)
    return jsonify(result)

# ===============================
# MAIN (for local testing only)
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
