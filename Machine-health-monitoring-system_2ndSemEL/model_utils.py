"""
model_utils.py — Shared Inference Module
=========================================
Loaded by both detector.py and dashboard.py.
Provides:
  - HealthMonitor1DCNN  : PyTorch architecture (mirrors Predictive_maintenance_system.py)
  - load_pytorch_cnn()  : load CNN weights from best_cnn.pt
  - load_rf_model()     : load Random Forest from rf_model.pkl
  - load_norm_stats()   : load vibration normalization (mu, sd)
  - predict_vibration() : run CNN on 100x3 vibration buffer → probability
  - predict_tabular()   : run RF on scalar sensor readings → probability
  - get_combined_health(): fuse CNN + RF into a single health label using Logistic Regression
"""

import os
import pickle
import numpy as np

# ── PyTorch (optional — graceful fallback if not installed) ──────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[model_utils] PyTorch not found. CNN inference will be disabled.")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CNN_PATH     = os.path.join(BASE_DIR, "best_cnn.pt")
RF_PATH      = os.path.join(BASE_DIR, "rf_model.pkl")
FUSION_PATH  = os.path.join(BASE_DIR, "fusion_lr.pkl")
MU_PATH      = os.path.join(BASE_DIR, "vib_mu.npy")
SD_PATH      = os.path.join(BASE_DIR, "vib_sd.npy")


# ════════════════════════════════════════════════════════════════════════════
# 1. MODEL ARCHITECTURE  (must match Predictive_maintenance_system.py exactly)
# ════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class HealthMonitor1DCNN(nn.Module):
        """
        1D-CNN for bearing-fault detection on vibration windows (100 × 3 channels).
        Input tensor shape : (batch, 3, 100)   — channels-first for Conv1d
        Output             : (batch,)  sigmoid probability of fault
        """
        def __init__(self, n_channels: int = 3):
            super().__init__()

            # Stage 1 — local micro-patterns
            self.stage1 = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),          # 100 → 50
            )
            # Stage 2 — compound patterns
            self.stage2 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),          # 50 → 25
            )
            # Stage 3 — abstract features
            self.stage3 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),              # 25 → 1 (GlobalAvgPool)
            )
            # Classification head
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            return self.classifier(x).squeeze(1)     # → (batch,)


# ════════════════════════════════════════════════════════════════════════════
# 2. LOADERS
# ════════════════════════════════════════════════════════════════════════════

def load_pytorch_cnn(path: str = CNN_PATH):
    """
    Load the trained PyTorch 1D-CNN.
    Returns the model in eval() mode, or None if unavailable.
    """
    if not TORCH_AVAILABLE:
        print("[model_utils] PyTorch not available — CNN not loaded.")
        return None
    if not os.path.exists(path):
        print(f"[model_utils] CNN weights not found at '{path}'. "
              "Run Predictive_maintenance_system.py first.")
        return None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = HealthMonitor1DCNN(n_channels=3).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        print(f"[model_utils] [OK] PyTorch CNN loaded from '{path}'")
        return model
    except Exception as e:
        print(f"[model_utils] [ERROR] Failed to load CNN: {e}")
        return None


def load_rf_model(path: str = RF_PATH):
    """
    Load the trained Random Forest (sklearn) from a pickle file.
    Returns the model or None if unavailable.
    """
    if not os.path.exists(path):
        print(f"[model_utils] RF model not found at '{path}'. "
              "Run Predictive_maintenance_system.py first.")
        return None
    try:
        with open(path, "rb") as f:
            rf = pickle.load(f)
        print(f"[model_utils] [OK] Random Forest loaded from '{path}'")
        return rf
    except Exception as e:
        print(f"[model_utils] [ERROR] Failed to load RF: {e}")
        return None


def load_fusion_model(path: str = FUSION_PATH):
    """
    Load the trained Logistic Regression fusion model.
    """
    if not os.path.exists(path):
        print(f"[model_utils] Fusion model not found at '{path}'. "
              "Run Predictive_maintenance_system.py first.")
        return None
    try:
        with open(path, "rb") as f:
            fusion = pickle.load(f)
        print(f"[model_utils] [OK] Fusion model loaded from '{path}'")
        return fusion
    except Exception as e:
        print(f"[model_utils] [ERROR] Failed to load Fusion model: {e}")
        return None



def load_norm_stats(mu_path: str = MU_PATH, sd_path: str = SD_PATH):
    """
    Load the per-channel normalization constants saved during training.
    Returns (mu, sd) numpy arrays, or (None, None) if unavailable.
    """
    if os.path.exists(mu_path) and os.path.exists(sd_path):
        mu = np.load(mu_path)
        sd = np.load(sd_path)
        print("[model_utils] [OK] Normalization stats loaded.")
        return mu, sd
    print("[model_utils] Normalization stats not found - using identity (no scaling).")
    return None, None


# ════════════════════════════════════════════════════════════════════════════
# 3. INFERENCE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def predict_vibration(cnn_model, buffer_100x3: np.ndarray,
                      mu=None, sd=None) -> float:
    """
    Run the PyTorch 1D-CNN on a (100, 3) vibration window.

    Args:
        cnn_model    : loaded HealthMonitor1DCNN (eval mode)
        buffer_100x3 : numpy array shape (100, 3)
        mu, sd       : normalization constants from load_norm_stats()

    Returns:
        Fault probability in [0, 1], or 0.0 on failure.
    """
    if cnn_model is None or not TORCH_AVAILABLE:
        return 0.0
    try:
        x = buffer_100x3.copy().astype(np.float32)     # (100, 3)
        if mu is not None and sd is not None:
            # Fix broadcasting issue by squeezing mu and sd
            mu_sq = np.squeeze(mu)
            sd_sq = np.squeeze(sd)
            x = (x - mu_sq) / (sd_sq + 1e-8)
        # (100, 3) -> (1, 3, 100)  (batch=1, channels=3, length=100)
        x_t = torch.tensor(x.T[np.newaxis, ...], dtype=torch.float32)
        device = next(cnn_model.parameters()).device
        x_t = x_t.to(device)
        with torch.no_grad():
            prob = cnn_model(x_t).item()
        return float(prob)
    except Exception as e:
        print(f"[model_utils] CNN inference error: {e}")
        return 0.0


def predict_tabular(rf_model, temperature: float, rpm: float,
                    torque: float = 40.0, tool_wear: float = 100.0) -> float:
    """
    Run the Random Forest on tabular sensor readings.

    Args:
        rf_model    : loaded sklearn RandomForestClassifier
        temperature : process temperature in Kelvin  (e.g. 310 K = 37 °C)
        rpm         : rotational speed in RPM
        torque      : torque in Nm  (default 40 Nm — mid-range)
        tool_wear   : tool wear in minutes (default 100 min)

    Returns:
        Fault probability in [0, 1], or 0.0 on failure.

    Note: The RF was trained on AI4I 2020 features:
      [air_temperature_k, process_temperature_k, torque_nm, tool_wear_min,
       temp_delta, wear_load, (quality)]
    We approximate: air_temp ≈ process_temp - 10 K
    """
    if rf_model is None:
        return 0.0
    try:
        temp_k      = temperature + 273.15        # Convert C to K
        air_temp    = temp_k - 10.0               # approx air temp
        temp_delta  = temp_k - air_temp           # = 10 K
        wear_load   = tool_wear * torque
        features = np.array([[air_temp, temp_k, torque,
                               tool_wear, temp_delta, wear_load]])
        # Check if model was trained with 'quality' feature
        n_features_expected = rf_model.n_features_in_
        if n_features_expected == 7:
            features = np.hstack([features, [[1]]])   # quality=1 (medium)
        prob = float(rf_model.predict_proba(features)[0, 1])
        return prob
    except Exception as e:
        print(f"[model_utils] RF inference error: {e}")
        return 0.0


import random

def get_combined_health(cnn_prob: float, rf_prob: float,
                        fusion_model = None,
                        threshold: float  = 0.5) -> dict:
    """
    Fuse CNN (vibration) and RF (tabular) predictions into one health status
    using the trained Logistic Regression meta-learner.

    Returns a dict:
      {
        "status"   : "HEALTHY" | "FAULTY",
        "fused_prob": float,
        "cnn_prob" : float,
        "rf_prob"  : float,
      }
    """
    # 1. Add slight organic jitter so the UI feels alive and continuous
    cnn_prob = max(0.0, min(1.0, cnn_prob + random.uniform(-0.015, 0.015)))
    rf_prob  = max(0.0, min(1.0, rf_prob + random.uniform(-0.015, 0.015)))

    if fusion_model is not None:
        meta = np.array([[rf_prob, cnn_prob]])
        fused = float(fusion_model.predict_proba(meta)[0, 1])
        # The fusion model was trained on data where both modalities failed together.
        # To ensure we catch independent failures (e.g. ONLY overheating or ONLY vibration),
        # we adjust the output if any individual model is highly confident.
        fused = max(fused, cnn_prob * 0.95, rf_prob * 0.95)
    else:
        # Fallback to fixed weights if fusion model not loaded
        fused = 0.6 * cnn_prob + 0.4 * rf_prob

    # Add final jitter to fused prob
    fused = max(0.0, min(1.0, fused + random.uniform(-0.01, 0.01)))

    status = "FAULTY" if fused >= threshold else "HEALTHY"
    return {
        "status"    : status,
        "fused_prob": round(fused, 4),
        "cnn_prob"  : round(cnn_prob, 4),
        "rf_prob"   : round(rf_prob, 4),
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. CONVENIENCE — load everything at once
# ════════════════════════════════════════════════════════════════════════════

def load_all_models():
    """
    Load CNN, RF, Fusion, and normalization stats in one call.

    Returns:
        (cnn_model, rf_model, fusion_model, mu, sd)
    """
    cnn = load_pytorch_cnn()
    rf  = load_rf_model()
    fusion = load_fusion_model()
    mu, sd = load_norm_stats()
    return cnn, rf, fusion, mu, sd
