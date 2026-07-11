"""
Forest Fire Detection - CNN Model Training
--------------------------------------------
Deep Learning model (transfer learning with MobileNetV2) that classifies
images captured from a camera (Raspberry Pi Camera / USB webcam) as:
    0 -> No Fire
    1 -> Fire

Dataset structure expected:
    dataset/
        train/
            fire/       (images containing fire/smoke)
            nofire/     (normal forest images)
        val/
            fire/
            nofire/

You can use any public "Forest Fire Dataset" (e.g. Kaggle) with this structure.

Install dependencies:
    pip install tensorflow opencv-python numpy matplotlib --break-system-packages
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ---------------------- Config ----------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DATASET_DIR = "dataset"          # change to your dataset path
MODEL_OUT = "fire_detection_model.h5"

# ---------------------- Data Generators ----------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

print("Class indices:", train_generator.class_indices)  # {'fire': 0, 'nofire': 1} or similar

# ---------------------- Model (Transfer Learning) ----------------------
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze base for initial training

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)  # binary: fire vs nofire

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------- Callbacks ----------------------
checkpoint = ModelCheckpoint(
    MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1
)
early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# ---------------------- Train ----------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
)

# ---------------------- Fine-tune (optional, improves accuracy) ----------------------
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[checkpoint, early_stop],
)

model.save(MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")
