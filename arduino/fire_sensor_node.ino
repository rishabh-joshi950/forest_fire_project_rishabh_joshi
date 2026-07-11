/*
  Forest Fire Detection & Prediction - Arduino Sensor Node
  -----------------------------------------------------------
  Reads:
    - DHT11/DHT22        -> Temperature (C), Humidity (%)
    - MQ2 Gas/Smoke Sensor -> Analog smoke level (0-1023)
    - Flame Sensor (IR)   -> Digital flame detection (LOW = flame detected)

  Sends CSV data over Serial (USB) to the Raspberry Pi every 2 seconds:
      temperature,humidity,smoke_level,flame_detected

  The Raspberry Pi reads this serial stream (see raspberry_pi/serial_reader.py),
  scales smoke_level to match the ANN training range, and feeds it to the
  deep learning model for risk prediction.

  Wiring:
    DHT11 data pin      -> D2
    MQ2 analog out (AO) -> A0
    Flame sensor OUT     -> D3
    Buzzer (local alarm) -> D8
    LED (status)         -> D9

  Library required: "DHT sensor library" by Adafruit (Install via Library Manager)
*/

#include <DHT.h>

#define DHTPIN 2
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

#define MQ2_PIN A0
#define FLAME_PIN 3
#define BUZZER_PIN 8
#define LED_PIN 9

const unsigned long SEND_INTERVAL = 2000; // ms
unsigned long lastSendTime = 0;

// Local threshold for immediate on-site alarm (independent of Pi's DL model)
const int SMOKE_ALARM_THRESHOLD = 600;

void setup() {
  Serial.begin(9600);
  dht.begin();
  pinMode(FLAME_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);

  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  Serial.println("temperature,humidity,smoke_level,flame_detected");
}

void loop() {
  unsigned long now = millis();

  if (now - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = now;

    float humidity = dht.readHumidity();
    float temperature = dht.readTemperature();
    int smokeLevel = analogRead(MQ2_PIN);          // 0-1023
    int flameState = digitalRead(FLAME_PIN);        // 0 = flame detected (module dependent)
    bool flameDetected = (flameState == LOW);

    if (isnan(humidity) || isnan(temperature)) {
      Serial.println("ERROR: Failed to read DHT sensor");
      return;
    }

    // ---- Send data as CSV over Serial ----
    Serial.print(temperature, 1);
    Serial.print(",");
    Serial.print(humidity, 1);
    Serial.print(",");
    Serial.print(smokeLevel);
    Serial.print(",");
    Serial.println(flameDetected ? 1 : 0);

    // ---- Local immediate alarm (does not depend on Raspberry Pi) ----
    if (flameDetected || smokeLevel > SMOKE_ALARM_THRESHOLD) {
      digitalWrite(BUZZER_PIN, HIGH);
      digitalWrite(LED_PIN, HIGH);
    } else {
      digitalWrite(BUZZER_PIN, LOW);
      digitalWrite(LED_PIN, LOW);
    }
  }
}
