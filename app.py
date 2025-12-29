import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
from flask_cors import CORS

MODEL_PATH = "coconut_disease_model .keras"
# Create an instance of the Flask class
app = Flask(__name__)

# Replace with your Vercel frontend URL
FRONTEND_URL = 'https://coconut-disease-detection.vercel.app'

# Configure CORS - allows specific origin only
CORS(app, 
     origins=[FRONTEND_URL, 'http://localhost:3000'],  # Add local dev URL
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     credentials=True)  # If using cookies/sessions

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

# Use the route() decorator to tell Flask what URL should trigger the function
@app.route("/api/predict", methods=["POST"])
def generate():
    data = request.get_json()
img_data = data.get("imageSrc")
    if not img_data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Decode the base64 image data
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Error decoding image: {str(e)}"}), 400

    prediction = predict(img)
    return jsonify(prediction)

