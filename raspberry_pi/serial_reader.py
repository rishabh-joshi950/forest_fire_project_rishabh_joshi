"""
Raspberry Pi - Serial Reader + ANN Fire Risk Prediction
------------------------------------------------------------
Reads sensor data streamed from the Arduino (over USB serial) and feeds it
into the trained ANN model (../ann_sensor_prediction/fire_risk_ann_model.h5)
to produce a live fire-risk probability.

Install dependencies on Raspberry Pi:
    pip install pyserial tensorflow joblib numpy --break-system-packages

Wiring: connect Arduino to Raspberry Pi via USB cable.
Find the correct serial port with: ls /dev/ttyUSB* /dev/ttyACM*
"""

import time
import serial
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ---------------------- Config ----------------------
SERIAL_PORT = "/dev/ttyACM0"   # change to /dev/ttyUSB0 if needed
BAUD_RATE = 9600
MODEL_PATH = "../ann_sensor_prediction/fire_risk_ann_model.h5"
SCALER_PATH = "../ann_sensor_prediction/sensor_scaler.pkl"

# Defaults for features not sent by this Arduino build (wind/rain sensors optional)
DEFAULT_WIND_SPEED = 5.0
DEFAULT_RAINFALL = 0.0

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def predict_risk(temperature, humidity, smoke_level, wind_speed, rainfall):
    features = np.array([[temperature, humidity, smoke_level, wind_speed, rainfall]])
    features_scaled = scaler.transform(features)
    prob = model.predict(features_scaled, verbose=0)[0][0]
    return float(prob)


def main():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
    time.sleep(2)  # allow Arduino to reset after serial connect
    print("Listening for sensor data... (Ctrl+C to stop)")

    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("temperature") or line.startswith("ERROR"):
                continue

            parts = line.split(",")
            if len(parts) != 4:
                continue

            temperature, humidity, smoke_raw, flame_flag = parts
            temperature = float(temperature)
            humidity = float(humidity)
            smoke_level = float(smoke_raw)   # Arduino analogRead range 0-1023
            flame_detected = int(flame_flag) == 1

            # Scale smoke_level (0-1023) to the training range (0-800) if needed
            smoke_level_scaled = smoke_level * (800 / 1023)

            prob = predict_risk(
                temperature, humidity, smoke_level_scaled,
                DEFAULT_WIND_SPEED, DEFAULT_RAINFALL,
            )

            status = "FLAME DETECTED (local sensor)" if flame_detected else ""
            print(
                f"Temp={temperature:.1f}C  Hum={humidity:.1f}%  "
                f"Smoke={smoke_level:.0f}  RiskProb={prob:.3f}  {status}"
            )

            if prob >= 0.7 or flame_detected:
                print("  >>> ALERT: HIGH FIRE RISK <<<")
                # TODO: trigger GPIO buzzer/LED, send MQTT/IoT alert, log to file, etc.

        except KeyboardInterrupt:
            print("Stopping...")
            break
        except Exception as e:
            print(f"Error reading serial data: {e}")


if __name__ == "__main__":
    main()
