# ================================================================
# MACHINE HEALTH MONITORING — Full Pipeline
# PyTorch : 1D-CNN on CWRU vibration windows
# Sklearn   : Random Forest + SVM on AI4I 2020 tabular data
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── PyTorch imports ─────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ── Sklearn imports ─────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (brier_score_loss, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ================================================================
# ██████████████████████████████████████████████████████████████
#   PART A — RF / SVM ON AI4I 2020  (Tabular: Temp, RPM, Torque)
# ██████████████████████████████████████████████████████████████
# ================================================================

print("\n" + "="*60)
print("  PART A — Tabular Branch (RF + SVM on AI4I 2020)")
print("="*60)

# ================================================================
# A1 — LOAD AI4I 2020
# ================================================================
ai4i = pd.read_csv('ai4i2020.csv')
df   = ai4i.copy()

# Flatten column names
df.columns = (df.columns.str.strip()
                         .str.lower()
                         .str.replace(r'[\[\]\s/]+', '_', regex=True)
                         .str.replace(r'[^\w]', '', regex=True))
print("Columns:", list(df.columns))
print("Raw columns:", list(df.columns))
print("Transformed columns:", list(df.columns))
#Transformed columns: ['udi', 'product_id', 'type', 'air_temperature_k_', 'process_temperature_k_', 'torque_nm_', 'tool_wear_min_', 'machine_failure', 'twf', 'hdf', 'pwf', 'osf', 'rnf']

# ================================================================
# A2 — FEATURE ENGINEERING
#
# Raw tabular features are already useful, but derived features
# capture domain physics:
#
#  temp_delta    → heat buildup = process_temp - air_temp
#                  Overheating precedes ~30% of motor failures
#
#  power         → RPM × Torque  (mechanical power input)
#                  Spikes in power under constant RPM = overload
#
#  wear_load     → tool_wear × torque
#                  Worn tool under heavy load = highest fault risk
#
#  rpm_stability → torque / RPM
#                  High ratio = motor straining to maintain speed
# ================================================================

df['temp_delta']   = df['process_temperature_k_'] - df['air_temperature_k_']
df['wear_load']    = df['tool_wear_min_']          * df['torque_nm_']

if 'type' in df.columns:
    df['quality'] = LabelEncoder().fit_transform(df['type'])

FEATURES = [
    'air_temperature_k_', 'process_temperature_k_', 'torque_nm_', 'tool_wear_min_',
    'temp_delta', 'wear_load'
]
if 'quality' in df.columns:
    FEATURES.append('quality')

TARGET = 'machine_failure'
X_tab  = df[FEATURES].values.astype(np.float32)
y_tab  = df[TARGET].values.astype(int)

print(f"\nTabular dataset : {X_tab.shape}")
print(f"Class balance   : {y_tab.sum()} fault / {(1-y_tab).sum()} healthy  "
      f"({y_tab.mean()*100:.1f}% fault rate)")


# ================================================================
# A3 — SPLIT
# ================================================================

X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_tab, y_tab, test_size=0.30, stratify=y_tab, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

print(f"Train {X_tr.shape[0]} | Val {X_val.shape[0]} | Test {X_te.shape[0]}")

# ================================================================
# A4 — RANDOM FOREST
#
# Key hyperparameter choices:
#   n_estimators=300  : diminishing returns after ~200; 300 is safe
#   max_features='sqrt': each split considers √(n_features) features
#                        → diversity between trees → lower variance
#   class_weight='balanced': AI4I has ~3.4% fault rate (imbalanced)
#                            balanced weights up-weight the minority
#   min_samples_leaf=2: prevents single-sample leaves (overfitting)
# ================================================================
print("\n── Random Forest ──────────────────────────────")

rf = RandomForestClassifier(
    n_estimators    = 300,
    max_features    = 'sqrt',
    min_samples_leaf= 2,
    class_weight    = 'balanced',
    n_jobs          = -1,
    random_state    = SEED
)

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_auc  = cross_val_score(rf, X_tr, y_tr, cv=cv,
                           scoring='roc_auc', n_jobs=-1)
print(f"  5-Fold CV AUC : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

rf.fit(X_tr, y_tr)
y_prob_rf = rf.predict_proba(X_te)[:, 1]
y_pred_rf = (y_prob_rf >= 0.5).astype(int)
auc_rf    = roc_auc_score(y_te, y_prob_rf)

print(f"  Test AUC      : {auc_rf:.4f}")
print(classification_report(y_te, y_pred_rf,
                             target_names=['Healthy', 'Faulty']))

# Feature importances
fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("  Top features:")
print(fi.to_string())

# ================================================================
# ██████████████████████████████████████████████████████████████
#   PART B — 1D-CNN ON CWRU VIBRATION  (PyTorch)
# ██████████████████████████████████████████████████████████████
# ================================================================
print("\n" + "="*60)
print("  PART B — Vibration Branch (1D-CNN on CWRU, PyTorch)")
print("="*60)

# ================================================================
# B1 — CWRU DATA LOADER
#
# CWRU files are .mat format. Two options:
#   Option A: Load real CWRU .mat files (recommended)
#   Option B: Use physics-informed synthetic data (shown below)
#
# For Option A, download from:
# kaggle.com/datasets/brjapon/cwru-bearing-datasets
# Then run: pip install scipy
# ================================================================
def load_cwru_mat(filepath: str, key: str = 'DE_time') -> np.ndarray:
    """
    Load one CWRU .mat file and return the drive-end vibration array.
    key options: 'DE_time' (drive end), 'FE_time' (fan end), 'BA_time' (base)
    """
    from scipy.io import loadmat
    mat  = loadmat(filepath)
    # CWRU keys have variable prefixes; find the right one
    data_key = [k for k in mat.keys() if key in k]
    if not data_key:
        raise KeyError(f"Key containing '{key}' not found. Keys: {list(mat.keys())}")
    return mat[data_key[0]].flatten().astype(np.float32)

def cwru_to_windows(signal: np.ndarray,
                    window: int = 100,
                    stride: int = 50) -> np.ndarray:
    """
    Slice a 1D vibration signal into overlapping windows.
    Each window becomes one training sample of shape (window, 1).
    Stride < window gives overlapping windows → more samples.

    For 3-channel input (X, Y, Z), call this per channel and stack:
        wx = cwru_to_windows(ax_signal)
        wy = cwru_to_windows(ay_signal)
        wz = cwru_to_windows(az_signal)
        X  = np.stack([wx, wy, wz], axis=2)   # (N, 100, 3)
    """
    starts  = range(0, len(signal) - window + 1, stride)
    windows = np.stack([signal[s:s+window] for s in starts])
    return windows[:, :, np.newaxis]   # (N, 100, 1)

# ================================================================
# B2 — PHYSICS-INFORMED SYNTHETIC CWRU SUBSTITUTE
#
# Use this if you don't have the .mat files yet.
# Matches CWRU statistics: 12 kHz sampling, 1797 RPM,
# bearing fault frequencies in the 100-400 Hz range.
# Replace with real CWRU data for the final submission.
# ================================================================
def make_cwru_synthetic(n_per_class: int = 1000,
                        timesteps: int  = 100,
                        n_channels: int = 3) -> tuple:
    t = np.linspace(0, timesteps / 12000, timesteps)   # 12 kHz sample rate
    X, y = [], []

    for _ in range(n_per_class):
        # ── Healthy: dominant single frequency, tiny noise ──
        f0   = np.random.uniform(1750, 1800) / 60   # ~30 Hz shaft freq
        amp  = np.random.uniform(0.2, 0.5)
        chs  = [amp * np.sin(2*np.pi*f0*t + np.random.uniform(0, 6.28))
                + np.random.normal(0, 0.02, timesteps)
                for _ in range(n_channels)]
        X.append(np.stack(chs, axis=1))
        y.append(0)

    for _ in range(n_per_class):
        # ── Faulty: shaft freq + bearing defect harmonics + impacts ──
        f0   = np.random.uniform(1750, 1800) / 60
        bpfi = f0 * np.random.uniform(5.4, 5.5)    # inner race defect freq
        amp  = np.random.uniform(0.6, 1.2)
        sig_base = (amp * np.sin(2*np.pi*f0*t)
                   + 0.4*amp * np.sin(2*np.pi*bpfi*t)       # defect harmonic
                   + 0.2*amp * np.sin(2*np.pi*2*bpfi*t))    # 2nd harmonic
        # Amplitude modulation (cage effect)
        sig_base *= 1 + 0.3*np.sin(2*np.pi*f0*0.4*t)
        # Impact spikes
        for pos in np.random.randint(0, timesteps, np.random.randint(2, 5)):
            idx      = np.arange(timesteps)
            sig_base += np.random.uniform(1.0, 2.5) * np.exp(
                -200*(idx-pos)**2/timesteps)
        noise = np.random.normal(0, 0.12, timesteps)
        chs   = [sig_base + noise + np.random.normal(0, 0.03, timesteps)
                 for _ in range(n_channels)]
        X.append(np.stack(chs, axis=1))
        y.append(1)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

X_vib, y_vib = make_cwru_synthetic(n_per_class=1000)
print(f"Vibration dataset : {X_vib.shape}  |  "
      f"{int(y_vib.sum())} faulty / {int((1-y_vib).sum())} healthy")

# Normalise (per-channel z-score)
mu_vib = X_vib.mean(axis=(0,1), keepdims=True)
sd_vib = X_vib.std(axis=(0,1),  keepdims=True) + 1e-8
X_vib  = (X_vib - mu_vib) / sd_vib

Xv_tr, Xv_tmp, yv_tr, yv_tmp = train_test_split(
    X_vib, y_vib, test_size=0.30, stratify=y_vib, random_state=SEED)
Xv_val, Xv_te, yv_val, yv_te = train_test_split(
    Xv_tmp, yv_tmp, test_size=0.50, stratify=yv_tmp, random_state=SEED)

# ================================================================
# B3 — PYTORCH DATASET & DATALOADER
#
# PyTorch requires wrapping data in a Dataset object.
# __len__  : tells DataLoader how many samples exist
# __getitem__: returns one (X, y) pair as tensors
# DataLoader then batches, shuffles, and handles memory efficiently.
# ================================================================
class VibrationDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # PyTorch Conv1d expects (batch, channels, length)
        # Our X is (N, timesteps, channels) → transpose to (N, channels, timesteps)
        self.X = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = VibrationDataset(Xv_tr,  yv_tr)
val_ds   = VibrationDataset(Xv_val, yv_val)
test_ds  = VibrationDataset(Xv_te,  yv_te)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=0)

# ================================================================
# B4 — 1D-CNN MODEL (PyTorch nn.Module)
#
# PyTorch CNN architecture mirrors the Keras version exactly,
# but expressed as a class. Key differences from Keras:
#
#   nn.Conv1d(in, out, kernel) — channels FIRST (N, C, L)
#   nn.BatchNorm1d             — operates on channel dim
#   nn.AdaptiveAvgPool1d(1)    — GlobalAveragePooling equivalent;
#                                 collapses temporal dim to 1
#   forward()                  — you write the pass explicitly;
#                                 no magic .fit() — full control
# ================================================================
class HealthMonitor1DCNN(nn.Module):
    def __init__(self, n_channels: int = 3):
        super().__init__()

        # Stage 1 — local micro-patterns (kernel spans 5 time-steps)
        self.stage1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),           # length: 100 → 50
        )

        # Stage 2 — compound patterns
        self.stage2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),           # 50 → 25
        )

        # Stage 3 — abstract features
        self.stage3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),               # 25 → 1 (GlobalAvgPool)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),                          # (batch, 128, 1) → (batch, 128)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x).squeeze(1)      # (batch,)

model_cnn = HealthMonitor1DCNN(n_channels=3).to(DEVICE)
print(f"\nCNN parameter count: {sum(p.numel() for p in model_cnn.parameters()):,}")

# ================================================================
# B5 — TRAINING LOOP (PyTorch)
#
# In PyTorch you write the training loop explicitly.
# This is more verbose than Keras but gives full control.
#
# Each epoch:
#   1. model.train()  → enables Dropout + BatchNorm training mode
#   2. Forward pass   → model(X) → predictions
#   3. Loss           → BCELoss(pred, true)
#   4. optimizer.zero_grad() → clear previous gradients
#   5. loss.backward()       → compute new gradients (backprop)
#   6. optimizer.step()      → update weights (Adam)
#   7. model.eval()   → disables Dropout, uses running BN stats
#   8. Validation     → no_grad() context saves memory
# ================================================================
criterion = nn.BCELoss()
optimizer = optim.Adam(model_cnn.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5)
old_lr = scheduler.optimizer.param_groups[0]['lr']
if scheduler.optimizer.param_groups[0]['lr'] != old_lr:
    print(f"Learning rate changed to: {scheduler.optimizer.param_groups[0]['lr']}")
EPOCHS = 50
best_val_loss= float('inf')
patience_ctr = 0
PATIENCE     = 10         # early stopping
history      = {'train_loss': [], 'val_loss': [],
                'train_acc':  [], 'val_acc':  []}

def run_epoch(loader, train: bool):
    model_cnn.train() if train else model_cnn.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = model_cnn(X_batch)
            loss  = criterion(preds, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct    += ((preds >= 0.5).float() == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, correct / total

print("\n── Training 1D-CNN ────────────────────────────")
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader,   train=False)

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(va_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(va_acc)
    scheduler.step(va_loss)
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:>3} | "
              f"Train Loss {tr_loss:.4f}  Acc {tr_acc:.4f} | "
              f"Val Loss {va_loss:.4f}  Acc {va_acc:.4f}")
    # Early stopping
    if va_loss < best_val_loss:
        best_val_loss = va_loss
        patience_ctr  = 0
        torch.save(model_cnn.state_dict(), 'best_cnn.pt')
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

# Reload best weights
model_cnn.load_state_dict(torch.load('best_cnn.pt', map_location=DEVICE))

# ================================================================
# B6 — EVALUATE CNN ON TEST SET
# ================================================================
def get_predictions(loader):
    model_cnn.eval()
    probs, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            p = model_cnn(X_batch.to(DEVICE))
            probs.extend(p.cpu().numpy())
            labels.extend(y_batch.numpy())
    return np.array(probs), np.array(labels)

y_prob_cnn, y_true_cnn = get_predictions(test_loader)
y_pred_cnn = (y_prob_cnn >= 0.5).astype(int)
auc_cnn    = roc_auc_score(y_true_cnn, y_prob_cnn)

print(f"\n  CNN Test AUC : {auc_cnn:.4f}")
print(classification_report(y_true_cnn, y_pred_cnn,
                             target_names=['Healthy', 'Faulty']))

# ================================================================
# PART D — FUSION LAYER
# Logistic Regression meta-learner combining RF + CNN outputs
# ================================================================

print("\n" + "="*60)
print("  PART D — Fusion Layer (Logistic Regression meta-learner)")
print("="*60)

# ================================================================
# D1 — BUILD THE FUSION CALIBRATION DATASET
#
# Strategy: generate N paired samples where each sample has BOTH
# a tabular feature vector (for RF) and a vibration window (for CNN),
# with a single shared fault label.
#
# Healthy samples → low-stress tabular values + clean vibration
# Faulty  samples → high-stress tabular values + disturbed vibration
#
# This simulates what your ESP32 system will do at inference:
# read all sensors simultaneously and ask both models for a verdict.
# ================================================================

N_FUSION    = 2000   # total calibration samples
N_HALF      = N_FUSION // 2
TIMESTEPS   = 100
N_CH        = 3

def make_healthy_tabular(n: int) -> np.ndarray:
    """
    Simulate healthy tabular readings.
    Low temperature delta, moderate torque, low wear.
    Matches the feature order in FEATURES list (no RPM):
    [air_temp, proc_temp, torque, tool_wear, temp_delta, wear_load, quality]
    """
    air_temp  = np.random.uniform(295, 300, n)          # K  — normal range
    proc_temp = air_temp + np.random.uniform(9, 11, n)  # process ~10K above air
    torque    = np.random.uniform(20, 35, n)             # Nm — light load
    wear      = np.random.uniform(0, 80, n)              # min — early life
    temp_delta= proc_temp - air_temp
    wear_load = wear * torque
    quality   = np.random.randint(0, 3, n).astype(float)

    return np.column_stack([
        air_temp, proc_temp, torque, wear,
        temp_delta, wear_load, quality
    ]).astype(np.float32)

def make_faulty_tabular(n: int) -> np.ndarray:
    """
    Simulate faulty tabular readings.
    High temperature delta, heavy torque, high wear.
    """
    air_temp  = np.random.uniform(298, 305, n)
    proc_temp = air_temp + np.random.uniform(12, 16, n) # excess heat buildup
    torque    = np.random.uniform(55, 80, n)             # Nm — overload
    wear      = np.random.uniform(180, 250, n)           # min — end of life
    temp_delta= proc_temp - air_temp
    wear_load = wear * torque
    quality   = np.random.randint(0, 3, n).astype(float)

    return np.column_stack([
        air_temp, proc_temp, torque, wear,
        temp_delta, wear_load, quality
    ]).astype(np.float32)

def make_healthy_vibration(n: int) -> np.ndarray:
    """Healthy: single clean sinusoid + tiny sensor noise."""
    t   = np.linspace(0, TIMESTEPS / 12000, TIMESTEPS)
    out = []
    for _ in range(n):
        f0  = np.random.uniform(1750, 1800) / 60
        amp = np.random.uniform(0.2, 0.5)
        chs = [amp * np.sin(2*np.pi*f0*t + np.random.uniform(0, 6.28))
               + np.random.normal(0, 0.02, TIMESTEPS)
               for _ in range(N_CH)]
        out.append(np.stack(chs, axis=1))
    return np.array(out, dtype=np.float32)

def make_faulty_vibration(n: int) -> np.ndarray:
    """Faulty: harmonics + amplitude modulation + impact spikes."""
    t   = np.linspace(0, TIMESTEPS / 12000, TIMESTEPS)
    out = []
    for _ in range(n):
        f0   = np.random.uniform(1750, 1800) / 60
        bpfi = f0 * np.random.uniform(5.4, 5.5)
        amp  = np.random.uniform(0.6, 1.2)
        sig  = (amp * np.sin(2*np.pi*f0*t)
               + 0.4*amp * np.sin(2*np.pi*bpfi*t)
               + 0.2*amp * np.sin(2*np.pi*2*bpfi*t))
        sig *= 1 + 0.3*np.sin(2*np.pi*f0*0.4*t)
        for pos in np.random.randint(0, TIMESTEPS, np.random.randint(2, 5)):
            idx = np.arange(TIMESTEPS)
            sig += np.random.uniform(1.0, 2.5) * np.exp(
                   -200*(idx-pos)**2/TIMESTEPS)
        noise = np.random.normal(0, 0.12, TIMESTEPS)
        chs   = [sig + noise + np.random.normal(0, 0.03, TIMESTEPS)
                 for _ in range(N_CH)]
        out.append(np.stack(chs, axis=1))
    return np.array(out, dtype=np.float32)

# Generate paired calibration samples
print("Generating fusion calibration dataset...")

tab_healthy = make_healthy_tabular(N_HALF)
tab_faulty  = make_faulty_tabular(N_HALF)
vib_healthy = make_healthy_vibration(N_HALF)
vib_faulty  = make_faulty_vibration(N_HALF)

tab_fusion  = np.vstack([tab_healthy, tab_faulty])
vib_fusion  = np.vstack([vib_healthy, vib_faulty])
y_fusion    = np.array([0]*N_HALF + [1]*N_HALF, dtype=int)

# Shuffle together
perm        = np.random.permutation(N_FUSION)
tab_fusion  = tab_fusion[perm]
vib_fusion  = vib_fusion[perm]
y_fusion    = y_fusion[perm]

print(f"  Fusion set shape  — tabular : {tab_fusion.shape}")
print(f"  Fusion set shape  — vibration: {vib_fusion.shape}")
print(f"  Labels            : {y_fusion.sum()} faulty / "
      f"{(y_fusion==0).sum()} healthy")

# ================================================================
# D2 — NORMALISE VIBRATION FOR CNN
#
# Use the SAME mean/std computed on the CNN training set (mu_vib,
# sd_vib). This is critical — applying a different normalisation
# at fusion time would corrupt the CNN's probability estimates.
# ================================================================

vib_fusion_n = (vib_fusion - mu_vib) / sd_vib

# ================================================================
# D3 — GET PROBABILITY OUTPUTS FROM BOTH TRAINED MODELS
#
# RF  → predict_proba on tabular features → scalar in [0, 1]
# CNN → forward pass on vibration window  → scalar in [0, 1]
#
# The two probabilities become a 2-column feature matrix.
# Each row = [prob_rf, prob_cnn] for one machine reading.
# ================================================================

print("\nCollecting model probability outputs...")

# ── RF probabilities ────────────────────────────────────────────
prob_rf_fusion = rf.predict_proba(tab_fusion)[:, 1]   # shape (2000,)

# ── CNN probabilities ───────────────────────────────────────────
# Transpose to (N, channels, timesteps) for PyTorch Conv1d
vib_tensor = torch.tensor(
    vib_fusion_n.transpose(0, 2, 1), dtype=torch.float32
)

model_cnn.eval()
prob_cnn_fusion = []
BATCH = 64

with torch.no_grad():
    for i in range(0, len(vib_tensor), BATCH):
        batch = vib_tensor[i:i+BATCH].to(DEVICE)
        out   = model_cnn(batch).cpu().numpy()
        prob_cnn_fusion.extend(out)

prob_cnn_fusion = np.array(prob_cnn_fusion)           # shape (2000,)

# ── Stack into 2-feature matrix ────────────────────────────────
X_fusion_meta = np.column_stack([prob_rf_fusion,
                                  prob_cnn_fusion])   # shape (2000, 2)

print(f"  Meta-feature matrix shape : {X_fusion_meta.shape}")
print(f"  RF   prob — mean: {prob_rf_fusion.mean():.3f}  "
      f"std: {prob_rf_fusion.std():.3f}")
print(f"  CNN  prob — mean: {prob_cnn_fusion.mean():.3f}  "
      f"std: {prob_cnn_fusion.std():.3f}")

# ================================================================
# D4 — SPLIT FUSION DATA AND TRAIN LOGISTIC REGRESSION
#
# Standard 80/20 split on the fusion calibration set.
# No stratified 3-way split needed here — only 2 features,
# tiny model, no risk of overfitting.
#
# C=1.0 : default regularisation, appropriate for 2 features.
# max_iter=1000 : ensures convergence on this clean data.
# ================================================================

X_fus_tr, X_fus_te, y_fus_tr, y_fus_te = train_test_split(
    X_fusion_meta, y_fusion,
    test_size=0.20, stratify=y_fusion, random_state=SEED
)

print(f"\n  Fusion train : {X_fus_tr.shape[0]} | "
      f"Fusion test : {X_fus_te.shape[0]}")

fusion_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
fusion_lr.fit(X_fus_tr, y_fus_tr)

# Show what weights were learned
print(f"\n  Learned weights:")
print(f"    RF  coefficient : {fusion_lr.coef_[0][0]:+.4f}")
print(f"    CNN coefficient : {fusion_lr.coef_[0][1]:+.4f}")
print(f"    Bias (intercept): {fusion_lr.intercept_[0]:+.4f}")
print(f"\n  Interpretation: a positive coefficient means the model")
print(f"  trusts that branch. Higher = more influential in decision.")

# ================================================================
# D5 — EVALUATE THE FULL FUSED SYSTEM
# ================================================================


y_fus_prob = fusion_lr.predict_proba(X_fus_te)[:, 1]
y_fus_pred = fusion_lr.predict(X_fus_te)
auc_fusion = roc_auc_score(y_fus_te, y_fus_prob)
brier      = brier_score_loss(y_fus_te, y_fus_prob)

print(f"\n── Fusion Layer Results ───────────────────────")
print(f"  AUC          : {auc_fusion:.4f}")
print(f"  Brier score  : {brier:.4f}  (0=perfect, 0.25=random)")
print(f"\n{classification_report(y_fus_te, y_fus_pred, target_names=['Working','Failure'])}")

# Compare all three AUC scores side by side
print("── AUC Comparison ─────────────────────────────")
print(f"  RF alone      : {auc_rf:.4f}")
print(f"  CNN alone     : {auc_cnn:.4f}")
print(f"  Fused system  : {auc_fusion:.4f}  ← should be ≥ both")

# ================================================================
# D6 — VISUALISE FUSION DECISION BOUNDARY
#
# Plot the 2D probability space with the learned boundary.
# Each point is one sample: x-axis = RF prob, y-axis = CNN prob.
# Colour = true label. Line = fusion model's decision boundary.
# ================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Fusion Layer — Full System Analysis', fontsize=13)

# ── Decision boundary plot ──────────────────────────────────────
ax = axes[0]
xx, yy = np.meshgrid(np.linspace(0, 1, 300), np.linspace(0, 1, 300))
grid   = np.column_stack([xx.ravel(), yy.ravel()])
zz     = fusion_lr.predict_proba(grid)[:, 1].reshape(xx.shape)

ax.contourf(xx, yy, zz, levels=50, cmap='RdYlGn_r', alpha=0.6)
ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=2)

scatter = ax.scatter(
    X_fus_te[:, 0], X_fus_te[:, 1],
    c=y_fus_te, cmap='RdYlGn_r', edgecolors='k',
    linewidths=0.4, s=30, alpha=0.8
)
ax.set_xlabel('RF Fault Probability',  fontsize=11)
ax.set_ylabel('CNN Fault Probability', fontsize=11)
ax.set_title('Fusion Decision Boundary\n(black line = 0.5 threshold)')
plt.colorbar(scatter, ax=ax, label='True label (0=Working, 1=Failure)')

# ── ROC curve comparison ────────────────────────────────────────
ax = axes[1]
for name, probs, true in [
    ("RF alone",     prob_rf_fusion[perm[:len(X_fus_te)]],  y_fus_te),
    ("CNN alone",    prob_cnn_fusion[perm[:len(X_fus_te)]], y_fus_te),
    ("Fused system", y_fus_prob,                            y_fus_te),
]:
    try:
        fpr, tpr, _ = roc_curve(true, probs)
        auc_val     = roc_auc_score(true, probs)
        lw          = 3 if name == "Fused system" else 1.5
        ls          = '-' if name == "Fused system" else '--'
        ax.plot(fpr, tpr, lw=lw, linestyle=ls,
                label=f"{name}  (AUC={auc_val:.3f})")
    except Exception:
        pass
ax.plot([0,1],[0,1],'k:', alpha=0.4)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Models')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── Confusion matrix — fused system ────────────────────────────
ax = axes[2]
cm = confusion_matrix(y_fus_te, y_fus_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Working', 'Failure'],
            yticklabels=['Working', 'Failure'],
            annot_kws={'size': 14})
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual',    fontsize=11)
ax.set_title('Fusion System\nConfusion Matrix')

plt.tight_layout()
plt.savefig('fusion_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ================================================================
# D7 — PRODUCTION INFERENCE FUNCTION
#
# This is the function your dashboard.py and serial_bridge.py
# will call at runtime. It takes:
#   tab_row  : 1D array of tabular sensor readings
#              [air_temp, proc_temp, torque, wear,
#               temp_delta, wear_load, quality]
#   vib_window: np.ndarray shape (100, 3) — raw vibration window
#
# Returns a dict with probability and human-readable label.
# ================================================================

def predict_machine_status(
    tab_row:    np.ndarray,
    vib_window: np.ndarray,
    threshold:  float = 0.5
) -> dict:
    """
    Full-system inference: RF + CNN → Fusion → binary verdict.

    Parameters
    ----------
    tab_row    : shape (7,)   — [air_temp, proc_temp, torque, wear,
                                  temp_delta, wear_load, quality]
    vib_window : shape (100,3) — normalised vibration window (X,Y,Z)
    threshold  : float         — decision boundary (default 0.5)

    Returns
    -------
    dict with individual model probs + final fused verdict
    """
    # ── Validate inputs ────────────────────────────────────────
    assert tab_row.shape    == (len(FEATURES),), \
        f"Expected tabular shape ({len(FEATURES)},), got {tab_row.shape}"
    assert vib_window.shape == (TIMESTEPS, N_CH), \
        f"Expected vibration shape ({TIMESTEPS},{N_CH}), got {vib_window.shape}"

    # ── RF inference ───────────────────────────────────────────
    p_rf = float(rf.predict_proba(tab_row.reshape(1, -1))[0, 1])

    # ── CNN inference ──────────────────────────────────────────
    # Normalise with training stats
    vib_n  = (vib_window - mu_vib.squeeze()) / sd_vib.squeeze()
    # Transpose to (1, channels, timesteps) for PyTorch
    vib_t  = torch.tensor(
        vib_n.T[np.newaxis], dtype=torch.float32
    ).to(DEVICE)

    model_cnn.eval()
    with torch.no_grad():
        p_cnn = float(model_cnn(vib_t).cpu().item())
     # ── Fusion inference ───────────────────────────────────────
    meta        = np.array([[p_rf, p_cnn]])
    p_fused     = float(fusion_lr.predict_proba(meta)[0, 1])
    label       = "⚠️  FAILURE" if p_fused >= threshold else "✅ WORKING"
    confidence  = max(p_fused, 1 - p_fused) * 100

    return {
        "verdict":          label,
        "fused_probability":round(p_fused, 4),
        "confidence":       f"{confidence:.1f}%",
        "rf_probability":   round(p_rf,    4),
        "cnn_probability":  round(p_cnn,   4),
    }

    # ── Demo call ──────────────────────────────────────────────────
demo_tab = make_faulty_tabular(1).flatten()
demo_vib = make_faulty_vibration(1)[0]
demo_vib = ((demo_vib - mu_vib.squeeze()) / sd_vib.squeeze())

result = predict_machine_status(demo_tab, demo_vib)

print("\n── Demo Inference ─────────────────────────────")
for k, v in result.items():
    print(f"  {k:<22}: {v}")

# ================================================================
# PART E — RESULTS DASHBOARD
# 2×3 grid: Row 1 = per-model performance
#            Row 2 = confusion matrices + fused ROC
# ================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Machine Health Monitoring — Full Results Dashboard',
             fontsize=14, fontweight='bold')
# ── [0,0] RF ROC on AI4I test set ──────────────────────────────
ax = axes[0, 0]
fpr, tpr, _ = roc_curve(y_te, y_prob_rf)
ax.plot(fpr, tpr, lw=2, color='steelblue',
        label=f"Random Forest  AUC={auc_rf:.3f}")
ax.plot([0,1],[0,1],'k--', alpha=0.3)
ax.set(title='ROC — Random Forest (AI4I)',
       xlabel='False Positive Rate', ylabel='True Positive Rate')
ax.legend(); ax.grid(alpha=0.3)
# ── [0,1] CNN training loss curve ──────────────────────────────
ax = axes[0, 1]
ax.plot(history['train_loss'], lw=2, label='Train loss', color='steelblue')
ax.plot(history['val_loss'],   lw=2, label='Val loss',
        linestyle='--', color='tomato')
ax.set(title='CNN Training — Loss', xlabel='Epoch', ylabel='BCE Loss')
ax.legend(); ax.grid(alpha=0.3)
# ── [0,2] CNN training accuracy curve ──────────────────────────
ax = axes[0, 2]
ax.plot(history['train_acc'], lw=2, label='Train acc', color='steelblue')
ax.plot(history['val_acc'],   lw=2, label='Val acc',
        linestyle='--', color='tomato')
ax.set(title='CNN Training — Accuracy', xlabel='Epoch', ylabel='Accuracy')
ax.legend(); ax.grid(alpha=0.3)
# ── [1,0] RF confusion matrix ──────────────────────────────────
ax = axes[1, 0]
cm = confusion_matrix(y_te, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Working','Failure'],
            yticklabels=['Working','Failure'],
            annot_kws={'size': 13})
ax.set(title='Confusion Matrix — Random Forest',
       xlabel='Predicted', ylabel='Actual')
# ── [1,1] CNN confusion matrix ─────────────────────────────────
ax = axes[1, 1]
cm = confusion_matrix(y_true_cnn, y_pred_cnn)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
            xticklabels=['Working','Failure'],
            yticklabels=['Working','Failure'],
            annot_kws={'size': 13})
ax.set(title='Confusion Matrix — 1D-CNN',
       xlabel='Predicted', ylabel='Actual')
# ── [1,2] Fusion confusion matrix — the headline result ────────
ax = axes[1, 2]
cm = confusion_matrix(y_fus_te, y_fus_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
            xticklabels=['Working','Failure'],
            yticklabels=['Working','Failure'],
            annot_kws={'size': 13})
ax.set(title=f'Confusion Matrix — Fused System\n(AUC={auc_fusion:.3f})',
       xlabel='Predicted', ylabel='Actual')
plt.tight_layout()
plt.savefig('results_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Final summary table ─────────────────────────────────────────
print("\n── Final Summary ──────────────────────────────")
print(f"{'Model':<22} {'Dataset':<12} {'AUC':>8}  {'Input'}")
print("─" * 62)
print(f"{'Random Forest':<22} {'AI4I 2020':<12} {auc_rf:>8.4f}  Tabular (temp, torque, wear)")
print(f"{'1D-CNN':<22} {'CWRU-style':<12} {auc_cnn:>8.4f}  Raw vibration waveform")
print(f"{'Fused System':<22} {'Both':<12} {auc_fusion:>8.4f}  RF prob + CNN prob → LR")

# --- NEW: Save Models for Real-time Inference ---
import pickle
print("\nSaving models for real-time inference...")
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('fusion_lr.pkl', 'wb') as f:
    pickle.dump(fusion_lr, f)
np.save('vib_mu.npy', mu_vib)
np.save('vib_sd.npy', sd_vib)
print("✅ Models saved: best_cnn.pt, rf_model.pkl, fusion_lr.pkl, vib_mu.npy, vib_sd.npy")
# ------------------------------------------------