"""
Fallback Fire Detector - Color & Brightness Heuristic
---------------------------------------------------------
Works on ANY image immediately, with no training required.
This is used automatically by the web dashboard when the trained
CNN model (fire_detection_model.h5) is not available yet.

How it works:
    Fire/flame regions are almost always bright pixels in the
    red -> orange -> yellow hue range. This function converts the
    image to HSV color space, masks pixels that fall in that range
    AND are bright/saturated enough to look like flame (not just
    orange-ish decor), then measures what fraction of the image
    those pixels cover.

This is a classical computer-vision technique (not deep learning),
so it's less accurate than a properly trained CNN on tricky images
(sunsets, orange clothing, autumn leaves can trigger false positives).
It's meant as a "works immediately" stand-in -- swap in train_cnn.py's
trained model for a real deep-learning result.
"""

import cv2
import numpy as np


def detect_fire_color(img_bgr):
    """
    img_bgr: image as a BGR numpy array (e.g. from cv2.imdecode / cv2.imread)
    Returns: (probability: float 0-1, fire_detected: bool)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Two hue ranges to catch red->orange->yellow flame tones
    lower1 = np.array([0, 90, 140])     # red-orange, must be saturated & bright
    upper1 = np.array([35, 255, 255])
    lower2 = np.array([0, 50, 200])     # near-white hot core of flame
    upper2 = np.array([35, 150, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up noise (isolated pixels shouldn't count)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    fire_pixel_ratio = float(np.count_nonzero(mask)) / mask.size

    # Scale the ratio into a 0-1 "probability"-like score.
    # Even a small but concentrated bright-flame patch should score high.
    probability = min(1.0, fire_pixel_ratio * 12)

    fire_detected = fire_pixel_ratio > 0.015  # >1.5% of image looks flame-colored

    return probability, fire_detected
