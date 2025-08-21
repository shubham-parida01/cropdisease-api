# microservice.py
import os
from io import BytesIO
import numpy as np
import cv2
import requests
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import tensorflow as tf

# =========================
# Configuration
# =========================
IMG_SIZE = 224
MODEL_PATH = "crop_disease_cnn_lstm_model.keras"
CLASS_NAMES_PATH = "class_names.npy"

# Load model and class names
model = tf.keras.models.load_model(MODEL_PATH)
raw_class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()

# =========================
# Human-readable class conversion
# =========================
def human_readable_class(class_name):
    parts = class_name.split('_', 1)
    crop = parts[0].capitalize()
    disease = parts[1].replace('_', ' ').title() if len(parts) > 1 else "Unknown"
    return crop, disease

# =========================
# Helper Functions
# =========================
def preprocess_image_array(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_disease_from_array(img, model, class_names):
    img = preprocess_image_array(img)
    preds = model.predict(img)
    top_idx = np.argmax(preds[0])
    confidence = float(preds[0][top_idx])
    predicted_class = class_names[top_idx]
    crop, disease = human_readable_class(predicted_class)
    return {"prediction": {"crop": crop, "disease": disease}, "confidence": confidence}

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Crop Disease Prediction API")

@app.post("/predict/file")
async def predict_from_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(content={"error": "Invalid image file"}, status_code=400)
        result = predict_disease_from_array(img, model, raw_class_names)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predict/url")
async def predict_from_url(image_url: str = Query(..., description="URL of the image")):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img_bytes = BytesIO(response.content)
        img_array = np.frombuffer(img_bytes.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(content={"error": "Could not read image from URL"}, status_code=400)
        result = predict_disease_from_array(img, model, raw_class_names)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "Crop Disease Prediction API is running!"}
