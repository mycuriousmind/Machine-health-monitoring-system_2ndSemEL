import streamlit as st
import numpy as np
import model_utils
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

# -- Load ML Models ------------------------------------------
@st.cache_resource
def load_all_models_cached():
    return model_utils.load_all_models()

cnn_model, rf_model, fusion_model, mu_vib, sd_vib = load_all_models_cached()

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
        background-color: #1e40af !important;
        border: 2px solid #3b82f6 !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .stMetric [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: 600 !important;
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

if cnn_model is None or rf_model is None:
    st.warning("⚠️ Some models could not be loaded. Please run `Predictive_maintenance_system.py` to train and save the PyTorch and RF models.")

# -- Sidebar Controls ----------------------------------------
with st.sidebar:
    st.header("Control Panel")
    sim_mode = st.radio("Data Source", ["Live from ESP32 Sensors", "Simulation"])
    sim_speed = st.slider("Simulation Speed (ms)", 100, 1000, 500)
    if sim_mode == "Simulation":
        force_battery_dead = st.checkbox("Force Battery Depleted")
    else:
        force_battery_dead = False
    # Battery will be calculated from current draw; no manual slider needed.
    
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
    
    st.info("This dashboard fuses a PyTorch 1D-CNN (vibration) and a Scikit-Learn Random Forest (telemetry).")
    
    st.markdown("---")
    st.subheader("Model Status")
    st.write(f"🧠 PyTorch CNN: {'✅ Loaded' if cnn_model else '❌ Missing'}")
    st.write(f"🌳 Random Forest: {'✅ Loaded' if rf_model else '❌ Missing'}")
    
    st.markdown("---")
    st.subheader("Battery Management")
    if st.button("🔋 Reset/Replace Battery", use_container_width=True):
        try:
            shared = get_shared_data() or {}
            shared["battery_remaining_mah"] = 500.0
            shared["last_charge_update_time"] = time.time()
            if "esp_data" in shared:
                shared["esp_data"]["Battery"] = 100.0
            with open("shared_data.json", "w") as f:
                json.dump(shared, f)
            st.success("Battery reset to 100%!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# -- Main Execution Loop -------------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []

shared_data = get_shared_data()

esp_disconnected = False

# 1. Gather Data based on mode
if sim_mode == "Live from ESP32 Sensors":
    esp_data = shared_data.get("esp_data", {}) if shared_data else {}
    esp_disconnected = (not esp_data or time.time() - esp_data.get("Timestamp", 0) > 5)
    
    if esp_disconnected:
        is_faulty = False
        prob = 0.0
        telemetry = {"Temperature": 0.0, "Vibration": 0.0, "RPM": 0.0, "Current": 0.0, "Signal": 0, "Battery": 0.0}
        cnn_prob = 0.0
        rf_prob = 0.0
        raw_data = np.zeros((100, 3))
    else:
        # Real ESP32 Data
        real_temp = esp_data.get("Temperature", 35.0)
        real_current = esp_data.get("Current", 0.0)
        real_battery = esp_data.get("Battery", 100.0)
        
        # Handle NaN values safely
        if real_temp is None or (isinstance(real_temp, float) and np.isnan(real_temp)):
            real_temp = 35.0
        if real_current is None or (isinstance(real_current, float) and np.isnan(real_current)):
            real_current = 0.0
        if real_battery is None or (isinstance(real_battery, float) and np.isnan(real_battery)):
            real_battery = 100.0
            
        # Physical Mappings: Current translates to Mechanical Load (Torque)
        mapped_torque = real_current * 20.0
        mapped_rpm = max(0, 3000.0 - (real_current * 100.0))
        
        # Check for real vibration buffer from ESP32, fallback to synthesis
        vibration_list = esp_data.get("VibrationBuffer")
        if vibration_list and len(vibration_list) == 100:
            raw_data = np.array(vibration_list, dtype=np.float32)
        else:
            # We synthesize vibration based on real mechanical state since ESP doesn't have an accelerometer yet
            is_esp_faulty = real_temp > 45.0 or real_current > 3.0
            
            t = np.linspace(0, 100 / 12000, 100)
            if is_esp_faulty:
                f0 = 30.0 
                bpfi = f0 * 5.4
                sig = 1.0 * np.sin(2*np.pi*f0*t) + 0.4 * np.sin(2*np.pi*bpfi*t) + 0.2 * np.sin(2*np.pi*2*bpfi*t)
                for pos in [20, 60]:
                    sig += 2.0 * np.exp(-200*(np.arange(100)-pos)**2/100)
                raw_data = np.stack([sig, sig, sig], axis=1).astype(np.float32) + np.random.randn(100, 3).astype(np.float32) * 0.2
            else:
                sig = 0.3 * np.sin(2*np.pi*30.0*t)
                raw_data = np.stack([sig, sig, sig], axis=1).astype(np.float32) + np.random.randn(100, 3).astype(np.float32) * 0.05
            
        # Inference using real mapped data
        cnn_prob = model_utils.predict_vibration(cnn_model, raw_data, mu_vib, sd_vib)
        rf_prob = model_utils.predict_tabular(rf_model, 
                                              temperature=real_temp,
                                              rpm=mapped_rpm,
                                              torque=mapped_torque,
                                              tool_wear=100.0) # Assume base wear for now
                                              
        health_res = model_utils.get_combined_health(cnn_prob, rf_prob, fusion_model)
        is_faulty = health_res["status"] == "FAULTY"
        prob = health_res["fused_prob"]
        
        telemetry = {
            "Temperature": real_temp,
            "Vibration": np.sqrt(np.mean(raw_data**2)),
            "RPM": mapped_rpm,
            "Current": real_current,
            "Signal": 100,
            "Battery": real_battery
        }
        # Determine if battery is dead (<=0%)
        battery_dead = telemetry["Battery"] <= 0
        # If battery is dead, treat motor as non-faulty
        if battery_dead:
            is_faulty = False
        
        
        # Save telemetry and health to shared_data.json so AR HUD updates in real-time
        try:
            shared = get_shared_data() or {}
            shared["telemetry"] = {
                "Temperature": real_temp,
                "Vibration": float(np.sqrt(np.mean(raw_data**2))),
                "RPM": mapped_rpm,
                "Signal": 100,
                "Core Temp": real_temp,
                "Current": real_current,
                "Battery": real_battery
            }
            shared["health"] = {
                "status": health_res["status"],
                "fused_prob": prob,
                "cnn_prob": cnn_prob,
                "rf_prob": rf_prob
            }
            shared["dashboard_timestamp"] = time.time()
            with open("shared_data.json", "w") as f:
                json.dump(shared, f)
        except:
            pass
    
else:
    # Simulation Mode
    # Initialize battery state (mAh) and timestamp if not already present
    if 'sim_battery_mah' not in st.session_state:
        st.session_state.sim_battery_mah = 1000.0  # increased capacity to slow discharge
        st.session_state.sim_battery_capacity = 1000.0  # store capacity for percent calc
        st.session_state.last_battery_update = time.time()
    t = np.linspace(0, 100 / 12000, 100)
    if force_vib:
        # Generate a faulty signal (sine + harmonics + impacts) similar to training
        f0 = 30.0 
        bpfi = f0 * 5.4
        sig = 1.0 * np.sin(2*np.pi*f0*t) + 0.4 * np.sin(2*np.pi*bpfi*t) + 0.2 * np.sin(2*np.pi*2*bpfi*t)
        for pos in [20, 60]:
            sig += 2.0 * np.exp(-200*(np.arange(100)-pos)**2/100)
        raw_data = np.stack([sig, sig, sig], axis=1).astype(np.float32) + np.random.randn(100, 3).astype(np.float32) * 0.2
    else:
        # Healthy signal (just small sine wave + noise)
        sig = 0.3 * np.sin(2*np.pi*30.0*t)
        raw_data = np.stack([sig, sig, sig], axis=1).astype(np.float32) + np.random.randn(100, 3).astype(np.float32) * 0.05
    
    # Run local prediction for CNN
    cnn_prob = model_utils.predict_vibration(cnn_model, raw_data, mu_vib, sd_vib)
    
    sim_current = random.uniform(3.5, 4.8) if (force_temp or force_rpm) else random.uniform(1.2, 2.2)
    # Update simulated battery based on current draw and elapsed time
    now = time.time()
    delta_h = (now - st.session_state.last_battery_update) / 3600  # hours elapsed since last update
    decrement_mah = sim_current * delta_h * 1000  # A * h = Ah, convert to mAh
    st.session_state.sim_battery_mah = max(0.0, st.session_state.sim_battery_mah - decrement_mah)
    st.session_state.last_battery_update = now
    capacity = getattr(st.session_state, 'sim_battery_capacity', 1000.0)
    battery_percent = (st.session_state.sim_battery_mah / capacity) * 100
    # If user forces battery depletion in simulation, set to 0
    if force_battery_dead:
        battery_percent = 0.0
    
    telemetry = {
        "Temperature": 95.0 if force_temp else random.uniform(34.0, 37.0),
        "Vibration": np.sqrt(np.mean(raw_data**2)),
        "RPM": 8500 if force_rpm else random.randint(2800, 3200),
        "Current": sim_current,
        "Signal": 95,
        "Core Temp": 95.0 if force_temp else random.uniform(34.0, 37.0),
        "Battery": battery_percent
    }
    
    rf_prob = model_utils.predict_tabular(rf_model,
                                          temperature=telemetry["Temperature"],
                                          rpm=telemetry["RPM"],
                                          torque=80.0 if (force_temp or force_rpm) else 40.0,
                                          tool_wear=220.0 if (force_temp or force_rpm) else 100.0)
                                          
    health_res = model_utils.get_combined_health(cnn_prob, rf_prob, fusion_model)
    is_faulty = health_res["status"] == "FAULTY"
    prob = health_res["fused_prob"]
    battery_dead = battery_percent <= 0
    # If battery dead, treat motor as non-faulty
    if battery_dead:
        is_faulty = False    
    # Save simulated telemetry so AR HUD matches exactly
    try:
        shared = get_shared_data() or {}
        shared["telemetry"] = telemetry
        shared["health"] = {
            "status": health_res["status"],
            "fused_prob": prob,
            "cnn_prob": cnn_prob,
            "rf_prob": rf_prob
        }
        shared["dashboard_timestamp"] = time.time()
        with open("shared_data.json", "w") as f:
            json.dump(shared, f)
    except: pass

# 3. Update History
st.session_state.history.append(prob)
if len(st.session_state.history) > 50:
    st.session_state.history.pop(0)

# 4. Render Interface
if esp_disconnected:
    st.markdown("<br><br><br><h1 style='text-align: center; color: red;'>⚠️ ESP32 DISCONNECTED!</h1><h2 style='text-align: center;'>PLEASE CONNECT ESP32</h2>", unsafe_allow_html=True)
    time.sleep(sim_speed / 1000)
    st.rerun()

# Status Card
status_class = "faulty" if is_faulty else "healthy"
# Override status if battery dead
if battery_dead:
    status_class = "faulty"
    status_text = "🔋 BATTERY DEPLETED"
else:
    status_text = "⚠️ ALERT: FAULT DETECTED" if is_faulty else "✅ SYSTEM HEALTHY"

st.markdown(f'<div class="status-card {status_class}">{status_text}</div>', unsafe_allow_html=True)

# Metrics Row
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Combined AI Confidence", f"{prob*100:.1f}%", delta=None)
m2.metric("CNN (Vibration)", f"{cnn_prob*100:.1f}%", delta=None)
m3.metric("RF (Telemetry)", f"{rf_prob*100:.1f}%", delta=None)
m4.metric("Temperature", f"{telemetry.get('Temperature', 0.0):.1f} °C", delta=None)
# Show zero current when battery depleted
display_current = 0.0 if battery_dead else telemetry.get('Current', 0.0)
m5.metric("Current", f"{display_current:.2f} A", delta=None)


# Charts Row
c1, c2 = st.columns([2, 1])

with c1:
    fig_wave = go.Figure()
    for axis, color in zip(range(3), ['#007bff', '#28a745', '#dc3545']):
        fig_wave.add_trace(go.Scatter(y=raw_data[:, axis], name=f"Axis {axis}", line=dict(color=color, width=1)))
    fig_wave.update_layout(title="Real-time Vibration (X, Y, Z)", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_wave, use_container_width=True)

with c2:
    cg1, cg2 = st.columns(2)
    with cg1:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = prob * 100,
            gauge = {'bar': {'color': "#ff0000" if is_faulty else "#00ff00"}, 'steps': [{'range': [0, 50], 'color': '#003300'}, {'range': [50, 100], 'color': '#330000'}]},
            title = {'text': "AI Confidence", 'font': {'size': 16}}
        ))
        fig_gauge.update_layout(template="plotly_dark", height=200, margin=dict(l=30, r=30, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with cg2:
        bat_val = telemetry.get("Battery", 100.0)
        fig_battery = go.Figure(go.Indicator(
            mode = "gauge+number", value = bat_val,
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#10b981" if bat_val > 20 else "#ef4444"},
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(239, 68, 68, 0.2)'},
                    {'range': [20, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
                ]
            },
            title = {'text': "Battery (%)", 'font': {'size': 16}}
        ))
        fig_battery.update_layout(template="plotly_dark", height=200, margin=dict(l=30, r=30, t=50, b=20))
        st.plotly_chart(fig_battery, use_container_width=True)

    # History Chart
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(y=st.session_state.history, fill='tozeroy', line=dict(color='#00ff00' if not is_faulty else '#ff0000')))
    fig_hist.update_layout(title="Fault Probability History", template="plotly_dark", height=150, margin=dict(l=20, r=20, t=40, b=20), yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig_hist, use_container_width=True)

# Loop control using rerun
time.sleep(sim_speed / 1000)
st.rerun()
