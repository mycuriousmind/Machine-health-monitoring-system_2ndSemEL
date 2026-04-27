import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import time
import os
import json
import subprocess
import random
import sys

# -- Set Page Config -----------------------------------------
st.set_page_config(
    page_title="Machine Health Monitor AI",
    page_icon="🤖",
    layout="wide",
)

# -- Load ML Model -------------------------------------------
@st.cache_resource
def load_ml_model():
    model_path = "fault_detector_model.keras"
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
    return None

model = load_ml_model()

# -- Shared Data Helper --------------------------------------
def get_shared_data():
    if os.path.exists("shared_data.json"):
        try:
            with open("shared_data.json", "r") as f:
                return json.load(f)
        except:
            return None
    return None

def save_dashboard_fault_triggers(current_triggers):
    try:
        shared = get_shared_data() or {}
        shared["dashboard_faults"] = current_triggers
        with open("shared_data.json", "w") as f:
            json.dump(shared, f)
    except:
        pass

# -- Custom CSS for Premium Look -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0f1116;
    }
    .stMetric {
        background-color: #1e222d;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .status-card {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .healthy {
        background-color: rgba(0, 255, 0, 0.1);
        border: 2px solid #00ff00;
        color: #00ff00;
    }
    .faulty {
        background-color: rgba(255, 0, 0, 0.1);
        border: 2px solid #ff0000;
        color: #ff0000;
    }
    .ar-button {
        display: inline-block;
        padding: 0.5em 1em;
        background-color: #4CAF50;
        color: white;
        text-align: center;
        text-decoration: none;
        font-size: 16px;
        border-radius: 8px;
        margin: 10px 0px;
    }
    </style>
    """, unsafe_allow_html=True)

# -- Title and Header ----------------------------------------
st.title("🛡️ Machine Health AI Dashboard")
st.markdown("### Real-time Vibration Analysis & Fault Prediction")

# -- AR Control Button ---------------------------------------
col_header1, col_header2 = st.columns([3, 1])
with col_header2:
    if st.button("🚀 LAUNCH AR DETECTOR", use_container_width=True):
        subprocess.Popen([sys.executable, "detector.py"])
        st.success("AR Detector launched!")

st.divider()

if model is None:
    st.error("❌ Model file 'fault_detector_model.keras' not found. Please train the model first using 1D_CNN.py.")
    st.stop()

# -- Sidebar Controls ----------------------------------------
with st.sidebar:
    st.header("Control Panel")
    sim_mode = st.radio("Data Source", ["Simulation", "Live from AR Detector"])
    sim_speed = st.slider("Simulation Speed (ms)", 100, 1000, 500)
    
    st.markdown("---")
    st.subheader("Manual Fault Injection")
    force_temp = st.checkbox("Force Temperature Fault")
    force_vib = st.checkbox("Force Vibration Fault")
    force_rpm = st.checkbox("Force RPM Fault")
    
    # Sync faults with dashboard
    dashboard_faults = {
        "Temperature": force_temp,
        "Vibration": force_vib,
        "RPM": force_rpm,
        "Core Temp": force_temp,
        "Signal": False
    }
    save_dashboard_fault_triggers(dashboard_faults)
    
    st.info("This dashboard uses your trained 1D-CNN model to analyze vibration patterns.")

# -- Main Execution Loop -------------------------------------
placeholder = st.empty()

if 'history' not in st.session_state:
    st.session_state.history = []

while True:
    shared_data = get_shared_data()
    
    # 1. Gather Data based on mode
    if sim_mode == "Live from AR Detector" and shared_data:
        telemetry = shared_data.get("telemetry", {})
        prob = shared_data.get("health", {}).get("prob", 0.0)
        is_faulty = shared_data.get("health", {}).get("status", "") == "FAULTY"
        # Since we don't have the raw waveform from detector.py easily, we jitter around the reported values
        v_base = telemetry.get("Vibration", 0.0)
        raw_data = np.random.randn(100, 3).astype(np.float32) * (5.0 if is_faulty else 1.2) + v_base
    else:
        # Simulation Mode
        if force_vib:
            raw_data = np.random.randn(100, 3).astype(np.float32) * 5.0 
        else:
            raw_data = np.random.randn(100, 3).astype(np.float32) * 1.2
        
        # Run local prediction
        x_input = raw_data[np.newaxis, ...] 
        prob = float(model.predict(x_input, verbose=0)[0, 0])
        is_faulty = prob > 0.5
        
        telemetry = {
            "Temperature": 95.0 if force_temp else random.uniform(34.0, 37.0),
            "Vibration": np.sqrt(np.mean(raw_data**2)),
            "RPM": 8500 if force_rpm else random.randint(2800, 3200),
            "Signal": 95
        }
        # Save simulated telemetry so AR HUD matches exactly
        try:
            shared = get_shared_data() or {}
            shared["telemetry"] = telemetry
            with open("shared_data.json", "w") as f:
                json.dump(shared, f)
        except: pass
        
        import random # need for local sim

    # 3. Update History
    st.session_state.history.append(prob)
    if len(st.session_state.history) > 50:
        st.session_state.history.pop(0)

    # 4. Render Interface
    with placeholder.container():
        # Status Card
        status_class = "faulty" if is_faulty else "healthy"
        status_text = "⚠️ ALERT: FAULT DETECTED" if is_faulty else "✅ SYSTEM HEALTHY"
        st.markdown(f'<div class="status-card {status_class}">{status_text}</div>', unsafe_allow_html=True)
        
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Fault Probability", f"{prob*100:.1f}%", delta=None)
        m2.metric("Vibration RMS", f"{telemetry.get('Vibration', 0.0):.2f} g", delta=None)
        m3.metric("Temperature", f"{telemetry.get('Temperature', 0.0):.1f} °C", delta=None)

        # Charts Row
        c1, c2 = st.columns([2, 1])
        
        with c1:
            fig_wave = go.Figure()
            for axis, color in zip(range(3), ['#007bff', '#28a745', '#dc3545']):
                fig_wave.add_trace(go.Scatter(y=raw_data[:, axis], name=f"Axis {axis}", line=dict(color=color, width=1)))
            fig_wave.update_layout(title="Real-time Vibration (X, Y, Z)", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_wave, use_container_width=True, key="vibration_chart")

        with c2:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = prob * 100,
                gauge = {'bar': {'color': "#ff0000" if is_faulty else "#00ff00"}, 'steps': [{'range': [0, 50], 'color': '#003300'}, {'range': [50, 100], 'color': '#330000'}]},
                title = {'text': "AI Confidence", 'font': {'size': 18}}
            ))
            fig_gauge.update_layout(template="plotly_dark", height=300, margin=dict(l=30, r=30, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_chart")

        # History Chart
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(y=st.session_state.history, fill='tozeroy', line=dict(color='#00ff00' if not is_faulty else '#ff0000')))
        fig_hist.update_layout(title="Fault Probability History", template="plotly_dark", height=150, margin=dict(l=20, r=20, t=40, b=20), yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_hist, use_container_width=True, key="history_chart")

    time.sleep(sim_speed / 1000)

