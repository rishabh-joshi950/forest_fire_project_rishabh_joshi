"""
Forest Fire PREDICTION - ANN Model (Sensor Data)
--------------------------------------------------
This deep neural network predicts fire RISK PROBABILITY using
environmental sensor readings collected by the Arduino node:

    - temperature   (DHT11/DHT22)
    - humidity      (DHT11/DHT22)
    - smoke_level   (MQ2 gas/smoke sensor, analog 0-1023 or scaled 0-800)
    - wind_speed    (anemometer, optional)
    - rainfall      (rain sensor, optional)

Target: fire_occurred (0 = no fire risk, 1 = fire risk)

A synthetic dataset 'sample_sensor_data.csv' is provided for demo/testing.
Replace it with your real logged sensor data (same column names) for a
production model.

Install dependencies:
    pip install tensorflow scikit-learn pandas numpy joblib --break-system-packages
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------- Load Data ----------------------
DATA_PATH = "sample_sensor_data.csv"
df = pd.read_csv(DATA_PATH)

FEATURES = ["temperature", "humidity", "smoke_level", "wind_speed", "rainfall"]
TARGET = "fire_occurred"

X = df[FEATURES].values
y = df[TARGET].values

# ---------------------- Train/Test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------- Scaling ----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "sensor_scaler.pkl")  # needed later on Raspberry Pi for inference

# ---------------------- Build ANN ----------------------
model = Sequential([
    Dense(32, activation="relu", input_shape=(len(FEATURES),)),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# ---------------------- Train ----------------------
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.15,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
)

# ---------------------- Evaluate ----------------------
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# ---------------------- Save Model ----------------------
model.save("fire_risk_ann_model.h5")
print("Saved model -> fire_risk_ann_model.h5")
print("Saved scaler -> sensor_scaler.pkl")
