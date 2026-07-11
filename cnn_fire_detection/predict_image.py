"""
Forest Fire Detection - Single Image Prediction
-------------------------------------------------
Loads the trained CNN model and predicts whether a given image
(or camera frame) contains fire.

Usage:
    python predict_image.py path/to/image.jpg
"""

import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model

MODEL_PATH = "fire_detection_model.h5"
IMG_SIZE = (224, 224)
THRESHOLD = 0.5  # decision threshold

model = load_model(MODEL_PATH)


def preprocess_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_fire(img):
    """img: BGR numpy array (as read by cv2.imread or camera frame)"""
    processed = preprocess_image(img)
    prob = model.predict(processed, verbose=0)[0][0]
    label = "FIRE DETECTED" if prob >= THRESHOLD else "No Fire"
    return label, float(prob)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    label, confidence = predict_fire(frame)
    print(f"Prediction: {label}  |  Confidence: {confidence:.4f}")
