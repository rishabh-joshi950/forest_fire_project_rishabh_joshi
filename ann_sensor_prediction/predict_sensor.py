"""
Forest Fire PREDICTION - Real-time sensor inference
-------------------------------------------------------
Loads the trained ANN + scaler and predicts fire risk probability
for a new sensor reading (e.g. coming live from the Arduino over serial).

Usage:
    python predict_sensor.py <temperature> <humidity> <smoke_level> <wind_speed> <rainfall>

Example:
    python predict_sensor.py 38.5 22.0 610 15.2 0.0
"""

import sys
import numpy as np
import joblib
from tensorflow.keras.models import load_model

MODEL_PATH = "fire_risk_ann_model.h5"
SCALER_PATH = "sensor_scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def predict_fire_risk(temperature, humidity, smoke_level, wind_speed, rainfall):
    features = np.array([[temperature, humidity, smoke_level, wind_speed, rainfall]])
    features_scaled = scaler.transform(features)
    prob = model.predict(features_scaled, verbose=0)[0][0]
    risk_level = (
        "HIGH RISK - FIRE LIKELY" if prob >= 0.7
        else "MODERATE RISK" if prob >= 0.4
        else "LOW RISK"
    )
    return float(prob), risk_level


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python predict_sensor.py <temp> <humidity> <smoke> <wind> <rain>")
        sys.exit(1)

    temp, hum, smoke, wind, rain = map(float, sys.argv[1:6])
    prob, risk_level = predict_fire_risk(temp, hum, smoke, wind, rain)
    print(f"Fire Probability: {prob:.4f}")
    print(f"Risk Level: {risk_level}")
