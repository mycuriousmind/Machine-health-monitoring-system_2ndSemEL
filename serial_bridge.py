# serial_bridge.py
# Run this in a separate terminal: python serial_bridge.py
# It reads the ESP32's Serial output and writes to shared_data.json
# so that dashboard.py and detector.py see live sensor data.

import serial
import json
import time
import numpy as np
import re
from collections import deque
import keras

# ── CONFIG ─────────────────────────────────────────────────
SERIAL_PORT  = "COM3"       # Windows: COM3, COM4, etc.
                             # Linux/Mac: /dev/ttyUSB0 or /dev/cu.usbserial-...
BAUD_RATE    = 115200
WINDOW_SIZE  = 100           # must match CNN input
MODEL_PATH   = "fault_detector_model.keras"
JSON_PATH    = "shared_data.json"

# Normalisation constants — use values from YOUR training run
# (the train_mean and train_std saved in Block 2 of the training script)
TRAIN_MEAN   = 0.0
TRAIN_STD    = 1.0

# ── LOAD MODEL ─────────────────────────────────────────────
print("Loading CNN model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded. Connecting to serial port...")

# ── ROLLING WINDOW BUFFER ──────────────────────────────────
# We accumulate 100 readings before running inference.
# deque with maxlen automatically discards oldest reading
# when full — this gives us a sliding window with no manual
# index management.
buffer = deque(maxlen=WINDOW_SIZE)

# ── HELPERS ────────────────────────────────────────────────
def parse_esp32_line(line: str) -> dict | None:
    """
    Parse the ESP32 serial output format:
    'Vibration: 1.02 | Temp: 36.5°C | Current: 0.87'
    Returns a dict or None if the line is malformed.
    """
    try:
        vib  = float(re.search(r"Vibration:\s*([\d.]+)", line).group(1))
        temp = float(re.search(r"Temp:\s*([\d.]+)",      line).group(1))
        curr = float(re.search(r"Current:\s*([\-\d.]+)", line).group(1))
        return {"vibration": vib, "temp": temp, "current": curr}
    except (AttributeError, ValueError):
        return None


def run_cnn_inference(window_buffer) -> tuple[float, str]:
    """
    Converts the deque of readings into a (100, 3) numpy array,
    normalises it, and runs the CNN.
    Returns (probability, label).
    """
    # Stack: each item is [ax, ay, az] — shape (100, 3)
    arr = np.array(list(window_buffer), dtype=np.float32)
    arr = (arr - TRAIN_MEAN) / (TRAIN_STD + 1e-8)
    x   = arr[np.newaxis, ...]   # (1, 100, 3)
    prob  = float(model.predict(x, verbose=0)[0, 0])
    label = "FAULTY" if prob >= 0.5 else "HEALTHY"
    return prob, label


def write_shared_data(telemetry: dict, prob: float, label: str):
    """Atomically update shared_data.json."""
    payload = {
        "telemetry": telemetry,
        "health": {"prob": round(prob, 4), "status": label},
    }
    # Read existing keys (e.g. dashboard_faults) so we don't wipe them
    try:
        with open(JSON_PATH, "r") as f:
            existing = json.load(f)
        existing.update(payload)
        payload = existing
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    with open(JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)


# ── MAIN LOOP ──────────────────────────────────────────────
with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2) as ser:
    print(f"Connected to {SERIAL_PORT}. Listening...")
    while True:
        try:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw:
                continue

            parsed = parse_esp32_line(raw)
            if parsed is None:
                print(f"  [skip] {raw}")
                continue

            # The ESP32 gives us a scalar 'vibration' (RMS of all 3 axes).
            # We need to reconstruct approximate X, Y, Z from the raw
            # accelerometer. To do this properly, add a second Serial.print
            # in the ESP32 code that sends ax, ay, az directly (see note below).
            # For now we approximate with the scalar:
            vib_scalar = parsed["vibration"]
            ax_approx  = vib_scalar * 0.577   # equal distribution assumption
            ay_approx  = vib_scalar * 0.577
            az_approx  = vib_scalar * 0.577

            buffer.append([ax_approx, ay_approx, az_approx])

            telemetry = {
                "Temperature": parsed["temp"],
                "Vibration":   vib_scalar,
                "Current":     parsed["current"],
                "Signal":      95,
            }

            # Only run CNN when window is full
            if len(buffer) == WINDOW_SIZE:
                prob, label = run_cnn_inference(buffer)
                print(f"  CNN → {label} ({prob*100:.1f}%)  | Temp: {parsed['temp']}°C")
            else:
                prob, label = 0.0, "COLLECTING"
                print(f"  Buffer: {len(buffer)}/{WINDOW_SIZE} samples")

            write_shared_data(telemetry, prob, label)
            time.sleep(0.05)   # small sleep to avoid hammering the file

        except KeyboardInterrupt:
            print("\nBridge stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)