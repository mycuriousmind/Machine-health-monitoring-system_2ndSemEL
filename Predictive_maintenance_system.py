# ================================================================
# MACHINE HEALTH MONITORING — Full Pipeline
# PyTorch  : 1D-CNN on CWRU vibration windows
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
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
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
# Install first: pip install ucimlrepo
# ================================================================

from ucimlrepo import fetch_ucirepo

ai4i = fetch_ucirepo(id=601)
df   = pd.concat([ai4i.data.features, ai4i.data.targets], axis=1)

# Flatten column names
df.columns = (df.columns.str.strip()
                         .str.lower()
                         .str.replace(r'[\[\]\s/]+', '_', regex=True)
                         .str.replace(r'[^\w]', '', regex=True))
print("Columns:", list(df.columns))


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

df['temp_delta']   = df['process_temperature_k'] - df['air_temperature_k']
df['power']        = df['rotational_speed_rpm']  * df['torque_nm']
df['wear_load']    = df['tool_wear_min']          * df['torque_nm']
df['rpm_stability']= df['torque_nm'] / (df['rotational_speed_rpm'] + 1e-6)

if 'type' in df.columns:
    df['quality'] = LabelEncoder().fit_transform(df['type'])

FEATURES = [
    'air_temperature_k', 'process_temperature_k',
    'rotational_speed_rpm', 'torque_nm', 'tool_wear_min',
    'temp_delta', 'power', 'wear_load', 'rpm_stability',
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
# A5 — SVM  (RBF kernel)
#
# Key choices:
#   StandardScaler : SVM uses Euclidean distances → scale is critical
#                    RF does not need scaling (tree splits are ordinal)
#   C=10           : low regularisation → complex boundary → suits
#                    the non-linear fault regions in AI4I
#   gamma='scale'  : 1/(n_features × X.var()) — safe automatic choice
#   probability=True: activates Platt scaling so predict_proba() works
#                     Adds slight overhead but needed for AUC + fusion
# ================================================================

print("\n── SVM (RBF) ──────────────────────────────────")

svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='rbf', C=10.0, gamma='scale',
                   class_weight='balanced',
                   probability=True, random_state=SEED))
])

cv_auc_svm = cross_val_score(svm_pipe, X_tr, y_tr, cv=cv,
                              scoring='roc_auc', n_jobs=-1)
print(f"  5-Fold CV AUC : {cv_auc_svm.mean():.4f} ± {cv_auc_svm.std():.4f}")

svm_pipe.fit(X_tr, y_tr)
y_prob_svm = svm_pipe.predict_proba(X_te)[:, 1]
y_pred_svm = (y_prob_svm >= 0.5).astype(int)
auc_svm    = roc_auc_score(y_te, y_prob_svm)

print(f"  Test AUC      : {auc_svm:.4f}")
print(classification_report(y_te, y_pred_svm,
                             target_names=['Healthy', 'Faulty']))


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
    optimizer, mode='min', factor=0.5, patience=5, verbose=True)

EPOCHS       = 50
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
# PART C — RESULTS DASHBOARD
# ================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Machine Health Monitoring — Model Comparison', fontsize=14)

# ROC curves — tabular models
ax = axes[0, 0]
for name, y_prob in [("Random Forest", y_prob_rf), ("SVM", y_prob_svm)]:
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    ax.plot(fpr, tpr, lw=2, label=f"{name} AUC={roc_auc_score(y_te,y_prob):.3f}")
ax.plot([0,1],[0,1],'k--', alpha=0.3)
ax.set(title='ROC — Tabular (AI4I)', xlabel='FPR', ylabel='TPR')
ax.legend(); ax.grid(alpha=0.3)

# CNN learning curves
ax = axes[0, 1]
ax.plot(history['train_loss'], lw=2, label='Train loss')
ax.plot(history['val_loss'],   lw=2, label='Val loss', linestyle='--')
ax.set(title='CNN Loss Curve', xlabel='Epoch', ylabel='BCE Loss')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 2]
ax.plot(history['train_acc'], lw=2, label='Train acc')
ax.plot(history['val_acc'],   lw=2, label='Val acc', linestyle='--')
ax.set(title='CNN Accuracy Curve', xlabel='Epoch', ylabel='Accuracy')
ax.legend(); ax.grid(alpha=0.3)

# Confusion matrices
for ax, (name, y_pred, y_true) in zip(
    axes[1], [
        ("Random Forest", y_pred_rf,  y_te),
        ("SVM",           y_pred_svm, y_te),
        ("1D-CNN",        y_pred_cnn, y_true_cnn),
    ]):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['H','F'], yticklabels=['H','F'])
    ax.set(title=name, xlabel='Predicted', ylabel='Actual')

plt.tight_layout()
plt.savefig('results_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary
print("\n── Final Summary ──────────────────────────────")
print(f"{'Model':<20} {'Dataset':<12} {'Test AUC':>10}  Input type")
print("─" * 60)
print(f"{'Random Forest':<20} {'AI4I 2020':<12} {auc_rf:>10.4f}  Tabular (temp, RPM, torque)")
print(f"{'SVM (RBF)':<20} {'AI4I 2020':<12} {auc_svm:>10.4f}  Tabular (temp, RPM, torque)")
print(f"{'1D-CNN (PyTorch)':<20} {'CWRU-style':<12} {auc_cnn:>10.4f}  Raw vibration waveform")
