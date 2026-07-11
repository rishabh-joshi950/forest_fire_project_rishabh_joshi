# Forest Fire Prediction and Detection using Deep Neural Networks

A hybrid **Arduino + Raspberry Pi** IoT system that:
1. **Predicts** fire risk from environmental sensor data (ANN) — early warning, before fire starts.
2. **Detects** actual fire/smoke from live camera footage (CNN) — real-time confirmation.

This matches a typical embedded-systems / IoT course project lifecycle: planning →
schematic → Arduino firmware → Raspberry Pi deep learning → integration → testing → deployment.

## Project Structure

```
forest_fire_project/
├── arduino/
│   └── fire_sensor_node.ino        # Embedded C: DHT11 + MQ2 + Flame sensor -> Serial
├── ann_sensor_prediction/
│   ├── train_ann.py                # Trains ANN on sensor data (risk prediction)
│   ├── predict_sensor.py           # CLI inference on a single reading
│   └── sample_sensor_data.csv      # Synthetic demo dataset (replace with real logs)
├── cnn_fire_detection/
│   ├── train_cnn.py                # Trains CNN (MobileNetV2 transfer learning) on images
│   └── predict_image.py            # CLI inference on a single image
├── raspberry_pi/
│   ├── serial_reader.py            # Reads Arduino serial data -> runs ANN prediction
│   └── pi_realtime_detection.py    # Live camera feed -> runs CNN detection + GPIO alert
└── requirements.txt
```

## Hardware Components

| Component            | Purpose                              |
|-----------------------|---------------------------------------|
| Arduino Uno/Nano      | Sensor node, local alarm             |
| DHT11 / DHT22         | Temperature & Humidity               |
| MQ2 Gas/Smoke Sensor  | Smoke/gas concentration              |
| Flame (IR) Sensor     | Direct flame detection               |
| Buzzer + LED          | Local on-site alarm                  |
| Raspberry Pi 4        | Runs deep learning models            |
| Pi Camera / USB Webcam| Visual fire/smoke detection input    |
| USB cable             | Arduino <-> Raspberry Pi serial link |

## Software / Tools

- Arduino IDE (upload `fire_sensor_node.ino`)
- Raspberry Pi OS + Thonny/Terminal
- Python 3.9+, TensorFlow/Keras, OpenCV, scikit-learn

## Step-by-Step Workflow

### 1. Arduino setup
- Wire DHT11 -> D2, MQ2 -> A0, Flame sensor -> D3, Buzzer -> D8, LED -> D9.
- Install the **DHT sensor library** (Adafruit) via Arduino IDE Library Manager.
- Upload `arduino/fire_sensor_node.ino`.
- Open Serial Monitor (9600 baud) to confirm CSV output: `temperature,humidity,smoke_level,flame_detected`.

### 2. Train the ANN (sensor-based prediction)
```bash
cd ann_sensor_prediction
pip install -r ../requirements.txt --break-system-packages
python train_ann.py
```
This produces `fire_risk_ann_model.h5` and `sensor_scaler.pkl`.
Replace `sample_sensor_data.csv` with real logged sensor data for production use.

### 3. Train the CNN (image-based detection)
Collect/download a fire-image dataset arranged as:
```
dataset/train/fire, dataset/train/nofire, dataset/val/fire, dataset/val/nofire
```
```bash
cd cnn_fire_detection
python train_cnn.py
```
This produces `fire_detection_model.h5`.

### 4. Deploy on Raspberry Pi
- Connect Arduino via USB. Check port: `ls /dev/ttyACM*` or `/dev/ttyUSB*`.
- Run sensor-based prediction:
```bash
cd raspberry_pi
python serial_reader.py
```
- Run camera-based detection (separate terminal/process):
```bash
python pi_realtime_detection.py
```

### 5. Combine both signals
In a full deployment, run both scripts and trigger a **combined alert** (e.g., MQTT
message to a dashboard, ThingSpeak/IoT platform upload, or SMS) when either:
- ANN risk probability ≥ 0.7, OR
- CNN detects fire in the camera feed, OR
- Arduino's local flame sensor fires.

## Testing & Debugging Tips
- Validate ANN with `predict_sensor.py <temp> <humidity> <smoke> <wind> <rain>`.
- Validate CNN with `predict_image.py path/to/test_image.jpg`.
- Check serial wiring/baud rate mismatches if `serial_reader.py` gets no data.
- Log predictions with timestamps to CSV for your project report/thesis evaluation section.

## Web Dashboard (optional, easiest way to demo the project)

A ready-made web dashboard lives in `webapp/`. It shows a live risk gauge fed by
the ANN model, plus a photo-upload panel that runs the CNN.

```bash
pip install flask --break-system-packages   # (drop --break-system-packages on Windows)
cd webapp
python app.py        # or: py -3.11 app.py   on Windows
```

Then open **http://127.0.0.1:5000** in your browser. Move the sliders and click
"Read risk level" to see the gauge respond; upload a photo under "Visual
Confirmation" to test the CNN.

Requires `fire_risk_ann_model.h5` + `sensor_scaler.pkl` (from `train_ann.py`) and,
optionally, `fire_detection_model.h5` (from `train_cnn.py`) to already exist in
their respective folders.

## Possible Extensions (for creativity/marks)
- Push alerts to a Telegram bot or ThingSpeak IoT dashboard.
- Add a GPS module to tag exact fire location.
- Add a solar-powered supply circuit for remote forest deployment (power supply section of your syllabus).
- Use LSTM instead of plain ANN to model time-series sensor trends for earlier prediction.
