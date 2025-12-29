import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
from flask_cors import CORS

MODEL_PATH = "coconut_disease_model.keras"

app = Flask(__name__)

FRONTEND_URL = "https://coconut-disease-detection.vercel.app"

CORS(
    app,
    origins=[FRONTEND_URL, "http://localhost:3000"],
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Load model
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

@app.route("/api/predict", methods=["POST"])
def generate():
    data = request.get_json()
    img_data = data.get("imageSrc")

    if not img_data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    prediction = predict(img)
    return jsonify(prediction)

