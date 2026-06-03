# ============================================================
# 1D-CNN for Machine Health Monitoring (MPU6050 Vibration Data)
# Industry 4.0 | Binary Classification: Healthy vs. Faulty
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

# -- 0. Reproducibility --------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

# ============================================================
# BLOCK 1: DUMMY DATA GENERATOR
# Simulates MPU6050 sensor output for sanity-checking the model
# ============================================================

NUM_SAMPLES  = 64   # batch size
TIME_STEPS   = 100  # readings per sample
NUM_CHANNELS = 3    # X, Y, Z axes

# Simulate raw vibration signals (zero-mean, unit-variance noise)
X_dummy = np.random.randn(NUM_SAMPLES, TIME_STEPS, NUM_CHANNELS).astype(np.float32)

# Binary labels: 0 = Healthy, 1 = Faulty
y_dummy = np.random.randint(0, 2, size=(NUM_SAMPLES,)).astype(np.float32)

print("-- Dummy Data Shapes --------------------------")
print(f"  X shape : {X_dummy.shape}   ->  (batch, timesteps, channels)")
print(f"  y shape : {y_dummy.shape}   ->  (batch,)")
print(f"  Label distribution: {int(y_dummy.sum())} Faulty / {int((1-y_dummy).sum())} Healthy")


# ============================================================
# BLOCK 2: MODEL ARCHITECTURE
# ============================================================

model = keras.Sequential(name="HealthMonitor_1DCNN")

# -- Input ----------------------------------------------------
model.add(keras.Input(shape=(TIME_STEPS, NUM_CHANNELS)))   # (100, 3)

# -- Stage 1: Local Feature Extraction -----------------------
model.add(layers.Conv1D(
    filters     = 32,          # 32 learned pattern detectors
    kernel_size = 5,           # looks at 5 consecutive time-steps
    activation  = 'relu',
    padding     = 'same',      # output length == input length
    name        = 'conv1_local_patterns'
))
model.add(layers.BatchNormalization(name='bn1'))
model.add(layers.MaxPooling1D(pool_size=2, name='pool1'))  # (100→50)

# -- Stage 2: Higher-Level Pattern Extraction ----------------
model.add(layers.Conv1D(
    filters     = 64,
    kernel_size = 3,
    activation  = 'relu',
    padding     = 'same',
    name        = 'conv2_compound_patterns'
))
model.add(layers.BatchNormalization(name='bn2'))
model.add(layers.MaxPooling1D(pool_size=2, name='pool2'))  # (50→25)

# -- Stage 3: Abstract Feature Extraction --------------------
model.add(layers.Conv1D(
    filters     = 128,
    kernel_size = 3,
    activation  = 'relu',
    padding     = 'same',
    name        = 'conv3_abstract_features'
))
model.add(layers.BatchNormalization(name='bn3'))
model.add(layers.GlobalAveragePooling1D(name='gap'))       # (25,128)->(128,)

# -- Stage 4: Classification Head ----------------------------
model.add(layers.Dense(64, activation='relu', name='fc1'))
model.add(layers.Dropout(0.4, name='dropout'))             # regularisation
model.add(layers.Dense(1, activation='sigmoid', name='output_faulty_prob'))

model.summary()


# ============================================================
# BLOCK 3: COMPILE
# ============================================================

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-3),
    loss      = 'binary_crossentropy',
    metrics   = ['accuracy',
                 keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
)


# ============================================================
# BLOCK 4: SANITY-CHECK — FORWARD PASS ON DUMMY DATA
# ============================================================

print("\n-- Forward Pass Verification ------------------")
predictions = model.predict(X_dummy, verbose=0)
print(f"  Output shape : {predictions.shape}")           # (64, 1)
print(f"  Sample preds : {predictions[:5].flatten().round(4)}")
print(f"  Min / Max    : {predictions.min():.4f} / {predictions.max():.4f}")
assert predictions.shape == (NUM_SAMPLES, 1), "Shape mismatch — check architecture"
assert 0.0 <= predictions.min() and predictions.max() <= 1.0, "Sigmoid out of [0,1]"
print("Architecture verified — all assertions passed")


# ============================================================
# BLOCK 5: DEMO TRAINING LOOP (on dummy data)
# ============================================================

print("\n-- Training on Dummy Data (3 epochs) ----------")
history = model.fit(
    X_dummy, y_dummy,
    epochs          = 3,
    batch_size      = 16,
    validation_split= 0.2,
    verbose         = 1
)

# -- Save the Model ------------------------------------------
model.save("fault_detector_model.keras")
print("\nModel saved as 'fault_detector_model.keras'")


# ============================================================
# BLOCK 6: INFERENCE HELPER
# ============================================================

def predict_machine_health(raw_window: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Parameters
    ----------
    raw_window : np.ndarray  shape (100, 3)  — one sensor window
    threshold  : float       decision boundary (default 0.5)

    Returns
    -------
    dict with probability and human-readable label
    """
    assert raw_window.shape == (TIME_STEPS, NUM_CHANNELS), \
        f"Expected ({TIME_STEPS},{NUM_CHANNELS}), got {raw_window.shape}"

    x = raw_window[np.newaxis, ...]          # (1, 100, 3)
    prob = float(model.predict(x, verbose=0)[0, 0])
    label = "[FAULTY]" if prob >= threshold else "HEALTHY"
    return {"probability_faulty": round(prob, 4), "label": label}

# Test with one synthetic window
sample_window = np.random.randn(TIME_STEPS, NUM_CHANNELS).astype(np.float32)
result = predict_machine_health(sample_window)
print(f"\n-- Single Inference ----------------------------")
print(f"  {result}")