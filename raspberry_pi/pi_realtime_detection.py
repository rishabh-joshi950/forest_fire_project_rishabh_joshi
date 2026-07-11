"""
Raspberry Pi - Real-time Camera Fire Detection (CNN)
--------------------------------------------------------
Captures live video from the Raspberry Pi Camera Module or a USB webcam,
runs each frame through the trained CNN (../cnn_fire_detection/fire_detection_model.h5),
and raises a GPIO alert (buzzer/LED) when fire is detected.

Install dependencies on Raspberry Pi:
    pip install tensorflow opencv-python RPi.GPIO --break-system-packages

Wiring (optional local alert):
    Buzzer -> GPIO17
    LED    -> GPIO27
"""

import time
import numpy as np
import cv2
from tensorflow.keras.models import load_model

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("RPi.GPIO not available - running without hardware alert output.")

# ---------------------- Config ----------------------
MODEL_PATH = "../cnn_fire_detection/fire_detection_model.h5"
IMG_SIZE = (224, 224)
THRESHOLD = 0.5
BUZZER_PIN = 17
LED_PIN = 27
CAMERA_INDEX = 0  # 0 = default camera (works for USB webcam and Pi Camera via libcamera-v4l2)

model = load_model(MODEL_PATH)

if GPIO_AVAILABLE:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.setup(LED_PIN, GPIO.OUT)


def preprocess_frame(frame):
    resized = cv2.resize(frame, IMG_SIZE)
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)


def set_alert(active: bool):
    if GPIO_AVAILABLE:
        GPIO.output(BUZZER_PIN, GPIO.HIGH if active else GPIO.LOW)
        GPIO.output(LED_PIN, GPIO.HIGH if active else GPIO.LOW)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting real-time fire detection... Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            processed = preprocess_frame(frame)
            prob = model.predict(processed, verbose=0)[0][0]
            fire_detected = prob >= THRESHOLD

            label = f"FIRE DETECTED ({prob:.2f})" if fire_detected else f"No Fire ({prob:.2f})"
            color = (0, 0, 255) if fire_detected else (0, 255, 0)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            set_alert(fire_detected)

            cv2.imshow("Forest Fire Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        set_alert(False)
        if GPIO_AVAILABLE:
            GPIO.cleanup()


if __name__ == "__main__":
    main()
