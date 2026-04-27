# ================================================================
# FULL TRAINING PIPELINE — 1D-CNN Machine Health Monitor
# Google Colab | TensorFlow/Keras | Synthetic MPU6050 Data
# ================================================================

import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

tf.random.set_seed(42)
np.random.seed(42)

# ================================================================
# BLOCK 1 — PHYSICS-INFORMED SYNTHETIC DATA GENERATOR
#
# Strategy: simulate real sensor behaviour rather than pure noise.
#   HEALTHY  → low-amplitude, periodic, single-frequency vibration
#             (machine running smoothly at ~50 Hz)
#   FAULTY   → same base signal PLUS random impact spikes and
#             broadband noise (bearing wear / rotor imbalance)
# ================================================================

SAMPLE_RATE   = 1000   # Hz (MPU6050 can do up to 1 kHz)
TIME_STEPS    = 100    # samples per window
NUM_CHANNELS  = 3      # X, Y, Z axes
N_HEALTHY     = 500
N_FAULTY      = 500

def generate_healthy_window():
    """
    Smooth, periodic vibration. A well-balanced motor produces
    a near-sinusoidal signal at its rotation frequency.
    """
    t = np.linspace(0, TIME_STEPS / SAMPLE_RATE, TIME_STEPS)
    window = np.zeros((TIME_STEPS, NUM_CHANNELS))

    for ch in range(NUM_CHANNELS):
        base_freq  = np.random.uniform(45, 55)          # ~50 Hz rotation
        amplitude  = np.random.uniform(0.3, 0.6)        # low amplitude
        phase      = np.random.uniform(0, 2 * np.pi)
        noise      = np.random.normal(0, 0.02, TIME_STEPS)  # tiny sensor noise

        window[:, ch] = amplitude * np.sin(2 * np.pi * base_freq * t + phase) + noise

    return window.astype(np.float32)


def generate_faulty_window():
    """
    Degraded vibration. Faults add:
      - Amplitude modulation (bearing cage frequency)
      - Random impact spikes (chipped tooth / pitting)
      - Elevated broadband noise floor (general wear)
      - Harmonic distortion (non-linearity from looseness)
    """
    t = np.linspace(0, TIME_STEPS / SAMPLE_RATE, TIME_STEPS)
    window = np.zeros((TIME_STEPS, NUM_CHANNELS))

    for ch in range(NUM_CHANNELS):
        base_freq  = np.random.uniform(45, 55)
        amplitude  = np.random.uniform(0.8, 1.5)         # higher amplitude
        phase      = np.random.uniform(0, 2 * np.pi)

        # Base signal with harmonic distortion
        signal = (amplitude * np.sin(2 * np.pi * base_freq * t + phase)
                + 0.3 * np.sin(2 * np.pi * 2 * base_freq * t)    # 2nd harmonic
                + 0.15 * np.sin(2 * np.pi * 3 * base_freq * t))  # 3rd harmonic

        # Amplitude modulation (cage frequency ~10 Hz)
        cage_freq = np.random.uniform(8, 12)
        signal *= (1 + 0.4 * np.sin(2 * np.pi * cage_freq * t))

        # Random impact spikes (2–5 per window)
        n_spikes = np.random.randint(2, 6)
        spike_positions = np.random.randint(0, TIME_STEPS, n_spikes)
        spike_amplitudes = np.random.uniform(1.5, 3.0, n_spikes)
        for pos, amp in zip(spike_positions, spike_amplitudes):
            # Decaying impulse shape (realistic impact response)
            idx = np.arange(TIME_STEPS)
            signal += amp * np.exp(-50 * (idx - pos)**2 / TIME_STEPS)

        # Elevated broadband noise
        noise = np.random.normal(0, 0.15, TIME_STEPS)
        window[:, ch] = signal + noise

    return window.astype(np.float32)


# --- Generate Dataset ---
print("Generating synthetic vibration dataset...")

X_healthy = np.array([generate_healthy_window() for _ in range(N_HEALTHY)])
X_faulty  = np.array([generate_faulty_window()  for _ in range(N_FAULTY)])

X = np.concatenate([X_healthy, X_faulty], axis=0)   # (1000, 100, 3)
y = np.concatenate([np.zeros(N_HEALTHY), np.ones(N_FAULTY)], axis=0)  # 0=Healthy, 1=Faulty

print(f"Dataset shape : {X.shape}")
print(f"Labels        : {int((y==0).sum())} Healthy  |  {int((y==1).sum())} Faulty")


# ================================================================
# BLOCK 2 — NORMALISATION
#
# Compute mean and std on training split ONLY (data leakage prevention).
# Subtract mean → zero-centred; divide by std → unit variance.
# This prevents any single axis from dominating due to scale differences.
# ================================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Compute statistics from training data only
train_mean = X_train.mean(axis=(0, 1), keepdims=True)  # shape (1,1,3)
train_std  = X_train.std(axis=(0, 1),  keepdims=True) + 1e-8

X_train_n = (X_train - train_mean) / train_std
X_val_n   = (X_val   - train_mean) / train_std
X_test_n  = (X_test  - train_mean) / train_std

print(f"\nSplit sizes:")
print(f"  Train : {X_train_n.shape[0]} samples (70%)")
print(f"  Val   : {X_val_n.shape[0]}   samples (15%)")
print(f"  Test  : {X_test_n.shape[0]}  samples (15%)")


# ================================================================
# BLOCK 3 — MODEL ARCHITECTURE
# (same as previous response — reproduced for self-contained script)
# ================================================================

def build_model(time_steps=100, n_channels=3):
    model = keras.Sequential(name="HealthMonitor_1DCNN")
    model.add(keras.Input(shape=(time_steps, n_channels)))

    # Stage 1: local micro-patterns
    model.add(layers.Conv1D(32, kernel_size=5, activation='relu',
                            padding='same', name='conv1'))
    model.add(layers.BatchNormalization(name='bn1'))
    model.add(layers.MaxPooling1D(pool_size=2, name='pool1'))     # 100 → 50

    # Stage 2: compound patterns
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu',
                            padding='same', name='conv2'))
    model.add(layers.BatchNormalization(name='bn2'))
    model.add(layers.MaxPooling1D(pool_size=2, name='pool2'))     # 50 → 25

    # Stage 3: abstract features
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu',
                            padding='same', name='conv3'))
    model.add(layers.BatchNormalization(name='bn3'))
    model.add(layers.GlobalAveragePooling1D(name='gap'))          # 25×128 → 128

    # Classification head
    model.add(layers.Dense(64, activation='relu', name='fc1'))
    model.add(layers.Dropout(0.4, name='dropout'))
    model.add(layers.Dense(1, activation='sigmoid', name='output'))

    return model

model = build_model()
model.summary()


# ================================================================
# BLOCK 4 — COMPILE
# ================================================================

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-3),
    loss      = 'binary_crossentropy',
    metrics   = [
        'accuracy',
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
    ]
)


# ================================================================
# BLOCK 5 — CALLBACKS
#
# These fire automatically at the end of each epoch:
#
#  EarlyStopping    — stops training when val_loss stops improving
#                     for 10 consecutive epochs (patience=10).
#                     restore_best_weights=True means you get the
#                     best checkpoint, not the last one.
#
#  ReduceLROnPlateau— halves the learning rate if val_loss hasn't
#                     improved for 5 epochs. Helps escape plateaus
#                     without manually tuning a schedule.
#
#  ModelCheckpoint  — saves the best model weights to disk so you
#                     can reload without retraining.
# ================================================================

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='best_health_monitor.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
]


# ================================================================
# BLOCK 6 — TRAIN
# ================================================================

print("\nStarting training...")
history = model.fit(
    X_train_n, y_train,
    epochs          = 60,        # EarlyStopping will cut this short
    batch_size      = 32,
    validation_data = (X_val_n, y_val),
    callbacks       = callbacks,
    verbose         = 1
)


# ================================================================
# BLOCK 7 — LEARNING CURVES
# ================================================================

def plot_training_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Training History', fontsize=14, fontweight='bold')

    metrics = [
        ('loss',     'Loss',     'binary cross-entropy'),
        ('accuracy', 'Accuracy', ''),
        ('auc',      'AUC',      ''),
    ]

    for ax, (metric, title, ylabel) in zip(axes, metrics):
        ax.plot(history.history[metric],     label='Train', linewidth=2)
        ax.plot(history.history[f'val_{metric}'], label='Val', linewidth=2, linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel or metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_training_history(history)


# ================================================================
# BLOCK 8 — EVALUATE ON HELD-OUT TEST SET
# ================================================================

print("\n── Test Set Evaluation ───────────────────────────")
test_results = model.evaluate(X_test_n, y_test, verbose=0)
metric_names = ['loss', 'accuracy', 'auc', 'precision', 'recall']
for name, val in zip(metric_names, test_results):
    print(f"  {name:<12}: {val:.4f}")

# Derive F1 from precision and recall
prec    = test_results[metric_names.index('precision')]
rec     = test_results[metric_names.index('recall')]
f1      = 2 * (prec * rec) / (prec + rec + 1e-8)
print(f"  {'f1_score':<12}: {f1:.4f}")


# ================================================================
# BLOCK 9 — CONFUSION MATRIX + CLASSIFICATION REPORT
# ================================================================

y_pred_prob = model.predict(X_test_n, verbose=0).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['Healthy (0)', 'Faulty (1)']))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Healthy', 'Faulty'],
            yticklabels=['Healthy', 'Faulty'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix — Test Set')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()


# ================================================================
# BLOCK 10 — VISUALISE ONE HEALTHY VS ONE FAULTY WINDOW
# ================================================================

def plot_sample_windows():
    fig, axes = plt.subplots(2, 3, figsize=(13, 5))
    fig.suptitle('Sample Vibration Windows (after normalisation)',
                 fontsize=13, fontweight='bold')
    axis_labels = ['X-axis', 'Y-axis', 'Z-axis']
    colors      = ['steelblue', 'tomato']
    labels      = ['Healthy', 'Faulty']
    samples     = [X_train_n[y_train == 0][0],
                   X_train_n[y_train == 1][0]]

    for row, (sample, label, color) in enumerate(zip(samples, labels, colors)):
        for col, axis_name in enumerate(axis_labels):
            ax = axes[row, col]
            ax.plot(sample[:, col], color=color, linewidth=1)
            ax.set_title(f'{label} — {axis_name}', fontsize=11)
            ax.set_xlabel('Time step')
            ax.set_ylabel('Amplitude (norm.)')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sample_windows.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_sample_windows()


# ================================================================
# BLOCK 11 — THRESHOLD TUNING
#
# 0.5 is not always the best threshold. In industrial settings you
# usually want high Recall (catch every fault) at the cost of some
# false alarms. Plot precision-recall vs threshold to pick your
# operating point.
# ================================================================

from sklearn.metrics import precision_recall_curve

precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_prob)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(thresholds, precision_vals[:-1], label='Precision', linewidth=2)
ax.plot(thresholds, recall_vals[:-1],    label='Recall',    linewidth=2)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.6, label='Default threshold (0.5)')
ax.set_xlabel('Decision threshold')
ax.set_ylabel('Score')
ax.set_title('Precision vs. Recall at different thresholds')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('threshold_tuning.png', dpi=150, bbox_inches='tight')
plt.show()

# Example: pick threshold that achieves Recall >= 0.95
target_recall = 0.95
idx           = np.argmax(recall_vals[:-1] >= target_recall)
best_threshold = thresholds[idx]
print(f"\nThreshold for Recall ≥ {target_recall}: {best_threshold:.3f}")
print(f"  Precision at this threshold: {precision_vals[idx]:.3f}")


# ================================================================
# BLOCK 12 — SAVE + RELOAD + INFERENCE HELPER
# ================================================================

# Save full model
model.save('health_monitor_final.keras')
print("\nModel saved to health_monitor_final.keras")

# Reload (proves the save worked)
loaded_model = keras.models.load_model('health_monitor_final.keras')

def predict_health(window: np.ndarray,
                   mean: np.ndarray,
                   std:  np.ndarray,
                   threshold: float = 0.5) -> dict:
    """
    Production-ready inference function.

    Parameters
    ----------
    window    : np.ndarray  shape (100, 3)  — raw sensor reading
    mean      : train_mean  (saved alongside the model)
    std       : train_std   (saved alongside the model)
    threshold : float       (tune via Block 11 above)
    """
    assert window.shape == (TIME_STEPS, NUM_CHANNELS)
    x    = ((window - mean.squeeze()) / std.squeeze())[np.newaxis]  # (1,100,3)
    prob = float(loaded_model.predict(x, verbose=0)[0, 0])
    return {
        "probability_faulty": round(prob, 4),
        "label": "FAULTY" if prob >= threshold else "HEALTHY",
        "confidence": f"{max(prob, 1-prob)*100:.1f}%"
    }

# Demo
test_window = generate_faulty_window()
result = predict_health(test_window, train_mean, train_std, threshold=best_threshold)
print(f"\nDemo inference: {result}")
