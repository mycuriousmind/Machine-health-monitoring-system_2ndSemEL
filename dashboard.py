# ================================================================
# dashboard.py — FIXED VERSION
# Machine Health AI Dashboard | Streamlit + Plotly + 1D-CNN
# ================================================================

import streamlit as st
import numpy as np
import tensorflow as tf
import keras
import plotly.graph_objects as go
import time
import os
import json
import subprocess
import random
import sys
import tempfile

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Machine Health Monitor AI",
    page_icon="🤖",
    layout="wide",
)

# ── Normalisation constants ──────────────────────────────────
# ⚠️  Replace these with your actual training stats from Block 2
# of the training script (train_mean and train_std).
TRAIN_MEAN = 0.0
TRAIN_STD  = 1.0

# ── Load ML Model ────────────────────────────────────────────
@st.cache_resource
def load_ml_model():
    model_path = "fault_detector_model.h5"
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
    return None

model = load_ml_model()

# ── JSON Helpers (race-condition safe) ───────────────────────
def get_shared_data() -> dict:
    try:
        with open("shared_data.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def write_json_safe(path: str, data: dict):
    """Atomic write: temp file → os.replace (never leaves partial JSON)."""
    dir_ = os.path.dirname(os.path.abspath(path))
    with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        tmp_path = f.name
    os.replace(tmp_path, path)

def save_dashboard_fault_triggers(triggers: dict):
    shared = get_shared_data()
    shared["dashboard_faults"] = triggers
    write_json_safe("shared_data.json", shared)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #0f1116; }
.stMetric {
    background-color: #1e222d;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #30363d;
}
.status-card {
    padding: 30px; border-radius: 15px;
    text-align: center; font-size: 24px;
    font-weight: bold; margin-bottom: 20px;
}
.healthy { background-color: rgba(0,255,0,0.1); border: 2px solid #00ff00; color: #00ff00; }
.faulty  { background-color: rgba(255,0,0,0.1); border: 2px solid #ff0000; color: #ff0000; }
</style>
""", unsafe_allow_html=True)

# ── Title ────────────────────────────────────────────────────
st.title("🛡️ Machine Health AI Dashboard")
st.markdown("### Real-time Vibration Analysis & Fault Prediction")

# ── AR Detector Button (with process guard) ──────────────────
if "ar_process" not in st.session_state:
    st.session_state.ar_process = None

col_h1, col_h2 = st.columns([3, 1])
with col_h2:
    ar_alive = (
        st.session_state.ar_process is not None
        and st.session_state.ar_process.poll() is None
    )
    if ar_alive:
        if st.button("🛑 STOP AR DETECTOR", use_container_width=True):
            st.session_state.ar_process.terminate()
            st.session_state.ar_process = None
            st.rerun()
    else:
        if st.button("🚀 LAUNCH AR DETECTOR", use_container_width=True):
            st.session_state.ar_process = subprocess.Popen(
                [sys.executable, "detector.py"]
            )
            st.success("AR Detector launched!")

st.divider()

if model is None:
    st.error("❌ Model file 'fault_detector_model.keras' not found. Train the model first.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Control Panel")
    sim_mode  = st.radio("Data Source", ["Simulation", "Live from AR Detector"])
    sim_speed = st.slider("Update interval (ms)", 100, 1000, 500)
    st.markdown("---")
    st.subheader("Manual Fault Injection")
    force_temp = st.checkbox("Force Temperature Fault")
    force_vib  = st.checkbox("Force Vibration Fault")
    force_rpm  = st.checkbox("Force RPM Fault")
    st.info("Uses your trained 1D-CNN to analyse vibration patterns.")

save_dashboard_fault_triggers({
    "Temperature": force_temp,
    "Vibration":   force_vib,
    "RPM":         force_rpm,
    "Core Temp":   force_temp,
    "Signal":      False,
})

# ── Session State Init ───────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_update" not in st.session_state:
    st.session_state.last_update = 0.0

# ── Non-blocking rate limiter ────────────────────────────────
# If not enough time has passed since last update, just rerun
# immediately — keeps Streamlit responsive without blocking.
now = time.time()
if now - st.session_state.last_update < (sim_speed / 1000):
    time.sleep(0.05)   # tiny sleep to avoid 100% CPU spin
    st.rerun()
st.session_state.last_update = now

# ── Gather Data ──────────────────────────────────────────────
shared_data = get_shared_data()

if sim_mode == "Live from AR Detector" and shared_data.get("health"):
    telemetry = shared_data.get("telemetry", {})
    prob      = shared_data["health"].get("prob", 0.0)
    is_faulty = shared_data["health"].get("status", "") == "FAULTY"
    v_base    = telemetry.get("Vibration", 0.0)
    raw_data  = (np.random.randn(100, 3).astype(np.float32)
                 * (5.0 if is_faulty else 1.2) + v_base)
else:
    # Simulation mode
    scale    = 5.0 if force_vib else 1.2
    raw_data = np.random.randn(100, 3).astype(np.float32) * scale

    # ✅ FIX: normalise before inference
    raw_data_norm = (raw_data - TRAIN_MEAN) / (TRAIN_STD + 1e-8)
    x_input       = raw_data_norm[np.newaxis, ...]
    prob          = float(model.predict(x_input, verbose=0)[0, 0])
    is_faulty     = prob > 0.5
    telemetry     = {
        "Temperature": 95.0 if force_temp else random.uniform(34.0, 37.0),
        "Vibration":   float(np.sqrt(np.mean(raw_data ** 2))),
        "RPM":         8500  if force_rpm  else random.randint(2800, 3200),
        "Signal":      95,
    }
    # Write simulation telemetry so AR HUD stays in sync
    shared = get_shared_data()
    shared["telemetry"] = telemetry
    shared["health"]    = {"prob": round(prob, 4),
                           "status": "FAULTY" if is_faulty else "HEALTHY"}
    write_json_safe("shared_data.json", shared)

# ── History ──────────────────────────────────────────────────
st.session_state.history.append(prob)
if len(st.session_state.history) > 50:
    st.session_state.history.pop(0)

# ── Render UI ────────────────────────────────────────────────
status_class = "faulty" if is_faulty else "healthy"
status_text  = "⚠️ ALERT: FAULT DETECTED" if is_faulty else "✅ SYSTEM HEALTHY"
st.markdown(f'<div class="status-card {status_class}">{status_text}</div>',
            unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("Fault Probability", f"{prob*100:.1f}%")
m2.metric("Vibration RMS",     f"{telemetry.get('Vibration', 0.0):.2f} g")
m3.metric("Temperature",       f"{telemetry.get('Temperature', 0.0):.1f} °C")

c1, c2 = st.columns([2, 1])
with c1:
    fig_wave = go.Figure()
    for axis, color in zip(range(3), ['#007bff', '#28a745', '#dc3545']):
        fig_wave.add_trace(go.Scatter(
            y=raw_data[:, axis], name=f"Axis {axis}",
            line=dict(color=color, width=1)
        ))
    fig_wave.update_layout(title="Real-time Vibration (X, Y, Z)",
                           template="plotly_dark", height=300,
                           margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_wave, use_container_width=True, key="vibration_chart")

with c2:
    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = prob * 100,
        gauge = {
            'bar':   {'color': "#ff0000" if is_faulty else "#00ff00"},
            'steps': [{'range': [0,  50], 'color': '#003300'},
                      {'range': [50, 100], 'color': '#330000'}],
        },
        title = {'text': "AI Confidence", 'font': {'size': 18}}
    ))
    fig_gauge.update_layout(template="plotly_dark", height=300,
                            margin=dict(l=30, r=30, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_chart")

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    y=st.session_state.history, fill='tozeroy',
    line=dict(color='#ff0000' if is_faulty else '#00ff00')
))
fig_hist.update_layout(title="Fault Probability History",
                       template="plotly_dark", height=150,
                       margin=dict(l=20, r=20, t=40, b=20),
                       yaxis=dict(range=[0, 1]))
st.plotly_chart(fig_hist, use_container_width=True, key="history_chart")

# ── Trigger next cycle ───────────────────────────────────────
# This replaces the while True loop entirely.
# Streamlit re-runs the whole script from top, keeping the UI live.
st.rerun()
