"""
Forest Fire Monitoring - Web Dashboard (Flask backend)
--------------------------------------------------------
Serves a web dashboard that:
  1. Takes sensor readings (temperature, humidity, smoke, wind, rain) and
     returns a live fire-risk probability using the trained ANN model.
  2. Accepts an uploaded photo and runs it through the trained CNN to check
     for visible fire/smoke.

Run:
    py -3.11 app.py
Then open in your browser: http://127.0.0.1:5000
"""

import os
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------------------- Paths ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANN_MODEL_PATH = os.path.join(BASE_DIR, "..", "ann_sensor_prediction", "fire_risk_ann_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "..", "ann_sensor_prediction", "sensor_scaler.pkl")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "..", "cnn_fire_detection", "fire_detection_model.h5")

# ---------------------- Lazy-loaded models ----------------------
_ann_model = None
_scaler = None
_cnn_model = None


def get_ann_model():
    global _ann_model, _scaler
    if _ann_model is None:
        from tensorflow.keras.models import load_model
        _ann_model = load_model(ANN_MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
    return _ann_model, _scaler


def get_cnn_model():
    global _cnn_model
    if _cnn_model is None:
        from tensorflow.keras.models import load_model
        _cnn_model = load_model(CNN_MODEL_PATH)
    return _cnn_model


# ---------------------- Routes ----------------------
@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Sensor-based fire risk prediction (ANN)."""
    try:
        data = request.get_json()
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        smoke_level = float(data["smoke_level"])
        wind_speed = float(data["wind_speed"])
        rainfall = float(data["rainfall"])

        model, scaler = get_ann_model()
        features = np.array([[temperature, humidity, smoke_level, wind_speed, rainfall]])
        features_scaled = scaler.transform(features)
        prob = float(model.predict(features_scaled, verbose=0)[0][0])

        if prob >= 0.7:
            risk_level = "HIGH"
        elif prob >= 0.4:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        return jsonify({"probability": prob, "risk_level": risk_level})

    except FileNotFoundError:
        return jsonify({
            "error": "ANN model not found. Run train_ann.py in ann_sensor_prediction/ first."
        }), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/detect-image", methods=["POST"])
def detect_image():
    """Image-based fire detection. Uses the trained CNN if available,
    otherwise falls back to a color/brightness heuristic that works on
    any image with no training required."""
    try:
        import cv2
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not read image"}), 400

        try:
            model = get_cnn_model()
            resized = cv2.resize(img, (224, 224)).astype("float32") / 255.0
            batch = np.expand_dims(resized, axis=0)
            prob = float(model.predict(batch, verbose=0)[0][0])
            fire_detected = prob >= 0.5
            method = "cnn"
        except FileNotFoundError:
            from fire_color_detector import detect_fire_color
            prob, fire_detected = detect_fire_color(img)
            method = "color_fallback"

        return jsonify({
            "probability": prob,
            "fire_detected": fire_detected,
            "method": method,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
