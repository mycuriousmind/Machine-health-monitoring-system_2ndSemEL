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
from cloud_manager import CloudManager
from nodered_client import NodeREDClient
from influxdb_manager import InfluxDBManager
from back4app_manager import Back4AppManager

# -- Set Page Config -----------------------------------------
st.set_page_config(
    page_title="Machine Health Monitor AI",
    page_icon="🤖",
    layout="wide",
)

@st.cache_resource
def get_cloud_manager():
    return CloudManager()

@st.cache_resource
def get_nodered_client():
    return NodeREDClient()

@st.cache_resource
def get_influxdb_manager():
    return InfluxDBManager()

@st.cache_resource
def get_back4app_manager():
    return Back4AppManager()

cloud_mgr = get_cloud_manager()
nodered_client = get_nodered_client()
influx_mgr = get_influxdb_manager()
back4app_mgr = get_back4app_manager()

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

    st.markdown("---")
    st.subheader("☁️ Cloud Sync (ThingSpeak)")
    cfg = cloud_mgr.config
    cloud_enabled = st.checkbox("Enable Cloud Sync", value=cfg.get("enabled", False))
    write_key = st.text_input("Write API Key", value=cfg.get("write_api_key", ""), type="password")
    channel_id = st.text_input("Channel ID", value=cfg.get("channel_id", ""))
    read_key = st.text_input("Read API Key", value=cfg.get("read_api_key", ""), type="password")
    update_int = st.number_input("Sync Interval (s)", min_value=15, max_value=300, value=cfg.get("update_interval", 15))

    if st.button("Save Cloud Config", use_container_width=True):
        cloud_mgr.save_config(cloud_enabled, write_key, channel_id, read_key, update_int)
        st.success("Config saved!")
        time.sleep(0.5)
        st.rerun()

    status_colors = {
        "Connected & Syncing": "green",
        "Rate Limited / Update Rejected": "orange",
        "Network Error": "red",
        "Disabled": "gray",
        "Not Connected": "gray"
    }
    status_color = status_colors.get(cloud_mgr.last_status, "gray")
    st.markdown(f"**Status**: :{status_color}[{cloud_mgr.last_status}]")
    if cloud_mgr.last_error:
        st.caption(f"Error: {cloud_mgr.last_error}")

    # ── Node-RED Status ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔴 Node-RED")
    nr_status = nodered_client.last_status
    nr_color = "green" if "Connected" in nr_status else ("red" if "Error" in nr_status or "Failed" in nr_status else "gray")
    st.markdown(f"**Status**: :{nr_color}[{nr_status}]")
    if nodered_client.enabled:
        st.caption(f"URL: {nodered_client.base_url}")
        st.markdown(f"[Open Node-RED UI →]({nodered_client.base_url})")
    else:
        st.caption("Node-RED is disabled in integration_config.json")
    if nodered_client.last_error:
        st.caption(f"Error: {nodered_client.last_error}")

    # ── InfluxDB Status ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 InfluxDB")
    inf_status = influx_mgr.last_status
    inf_color = "green" if "Connected" in inf_status else ("red" if "Error" in inf_status else "gray")
    st.markdown(f"**Status**: :{inf_color}[{inf_status}]")
    if influx_mgr.enabled:
        inf_url = influx_mgr.config.get("url", "http://localhost:8086")
        st.caption(f"Bucket: {influx_mgr.config.get('bucket', 'telemetry')}")
        st.markdown(f"[Open InfluxDB UI →]({inf_url})")
    else:
        st.caption("InfluxDB is disabled in integration_config.json")
    if influx_mgr.last_error:
        st.caption(f"Error: {influx_mgr.last_error}")

    # ── Back4App Status ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("☁️ Back4App")
    b4a_status = back4app_mgr.last_status
    b4a_color = "green" if "Connected" in b4a_status else ("red" if "Error" in b4a_status else "gray")
    st.markdown(f"**Status**: :{b4a_color}[{b4a_status}]")
    if back4app_mgr.enabled:
        st.caption("Parse Server: Back4App Cloud")
    else:
        st.caption("Back4App is disabled in integration_config.json")
    if back4app_mgr.last_error:
        st.caption(f"Error: {back4app_mgr.last_error}")

    # ── Maintenance Log (Back4App) ───────────────────────────────
    st.markdown("---")
    st.subheader("🔧 Maintenance Log")
    maint_action = st.selectbox("Action", [
        "BATTERY_REPLACED", "CALIBRATION", "SENSOR_CLEANED",
        "SYSTEM_RESTART", "INSPECTION", "OTHER"
    ])
    maint_notes = st.text_input("Notes (optional)", value="")
    if st.button("📝 Log Maintenance", use_container_width=True):
        result = back4app_mgr.log_maintenance(action=maint_action, notes=maint_notes)
        if result:
            st.success(f"Logged: {maint_action} (ID: {result.get('objectId', 'N/A')})")
        else:
            st.warning("Could not log maintenance. Check Back4App config.")

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

# ═══ MULTI-PLATFORM CLOUD UPLOAD ═══════════════════════════════════════════
# Upload telemetry to ThingSpeak (handled and rate-limited automatically)
cloud_mgr.upload_telemetry(
    temp=telemetry.get("Temperature", 0.0),
    current=telemetry.get("Current", 0.0),
    vibration=telemetry.get("Vibration", 0.0),
    rpm=telemetry.get("RPM", 0.0),
    battery=telemetry.get("Battery", 100.0),
    fused_prob=prob,
    cnn_prob=cnn_prob,
    rf_prob=rf_prob
)

# Push to Node-RED (fans out to InfluxDB, Back4App, ThingSpeak via flow)
nodered_client.push_telemetry(
    temp=telemetry.get("Temperature", 0.0),
    current=telemetry.get("Current", 0.0),
    vibration=telemetry.get("Vibration", 0.0),
    rpm=telemetry.get("RPM", 0.0),
    battery=telemetry.get("Battery", 100.0),
    fused_prob=prob,
    cnn_prob=cnn_prob,
    rf_prob=rf_prob
)

# Direct write to InfluxDB (in addition to Node-RED routing, for reliability)
influx_mgr.write_telemetry(
    temp=telemetry.get("Temperature", 0.0),
    current=telemetry.get("Current", 0.0),
    vibration=telemetry.get("Vibration", 0.0),
    rpm=telemetry.get("RPM", 0.0),
    battery=telemetry.get("Battery", 100.0),
    fused_prob=prob,
    cnn_prob=cnn_prob,
    rf_prob=rf_prob
)

# Log fault events to Back4App when detected
if is_faulty:
    back4app_mgr.log_machine_event(
        event_type="FAULT_DETECTED",
        severity="CRITICAL" if prob >= 0.8 else "WARNING",
        description=f"Fused AI fault detected (confidence: {prob*100:.1f}%)",
        telemetry={
            "temperature": telemetry.get("Temperature", 0.0),
            "current": telemetry.get("Current", 0.0),
            "vibration": telemetry.get("Vibration", 0.0),
            "rpm": telemetry.get("RPM", 0.0),
            "battery": telemetry.get("Battery", 100.0),
        },
        ai_predictions={
            "fused_prob": prob,
            "cnn_prob": cnn_prob,
            "rf_prob": rf_prob
        }
    )

# ═══ HISTORICAL ANALYTICS (InfluxDB) ══════════════════════════════════════
st.divider()
st.markdown("### 📈 Historical Analytics (InfluxDB)")

ha1, ha2 = st.columns(2)

with ha1:
    # Health history from InfluxDB
    health_history = influx_mgr.query_health_history(hours=24)
    if health_history:
        fig_health = go.Figure()
        times = [r["time"] for r in health_history]
        fig_health.add_trace(go.Scatter(
            x=times, y=[r["fused_prob"] for r in health_history],
            name="Fused Prob", fill="tozeroy",
            line=dict(color="#ef4444", width=2)
        ))
        fig_health.add_trace(go.Scatter(
            x=times, y=[r["cnn_prob"] for r in health_history],
            name="CNN Prob", line=dict(color="#3b82f6", width=1, dash="dot")
        ))
        fig_health.add_trace(go.Scatter(
            x=times, y=[r["rf_prob"] for r in health_history],
            name="RF Prob", line=dict(color="#10b981", width=1, dash="dot")
        ))
        fig_health.update_layout(
            title="24h Fault Probability Trend (InfluxDB)",
            template="plotly_dark", height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(range=[0, 1], title="Probability"),
            xaxis=dict(title="Time")
        )
        st.plotly_chart(fig_health, use_container_width=True)
    else:
        st.info("No historical data in InfluxDB yet. Data will appear after InfluxDB is running and receiving telemetry.")

with ha2:
    # Statistics from InfluxDB
    stats = influx_mgr.get_statistics(hours=24)
    if stats:
        st.markdown("**24h Statistics**")
        stats_data = []
        for field, values in stats.items():
            stats_data.append({
                "Metric": field.replace("_", " ").title(),
                "Min": f"{values['min']:.2f}",
                "Max": f"{values['max']:.2f}",
                "Mean": f"{values['mean']:.2f}",
                "Points": values['count']
            })
        st.dataframe(stats_data, use_container_width=True, hide_index=True)
    else:
        st.info("Statistics will be available once InfluxDB has data.")

# ═══ CLOUD EVENT LOG (Back4App) ═══════════════════════════════════════════
st.divider()
st.markdown("### 🔔 Cloud Event Log (Back4App)")

el1, el2 = st.columns([2, 1])

with el1:
    events = back4app_mgr.get_recent_events(limit=10)
    if events:
        event_display = []
        for evt in events:
            severity = evt.get("severity", "INFO")
            icon = "🔴" if severity == "CRITICAL" else ("🟡" if severity == "WARNING" else "🟢")
            event_display.append({
                "": icon,
                "Type": evt.get("eventType", "N/A"),
                "Severity": severity,
                "Description": evt.get("description", "")[:60],
                "Time": evt.get("createdAt", "N/A")[:19].replace("T", " ")
            })
        st.dataframe(event_display, use_container_width=True, hide_index=True)
    else:
        st.info("No events logged yet. Fault events will appear when the AI detects anomalies.")

with el2:
    maint = back4app_mgr.get_maintenance_history(limit=5)
    if maint:
        st.markdown("**Recent Maintenance**")
        for m in maint:
            st.write(f"🔧 **{m.get('action', 'N/A')}** — {m.get('notes', 'No notes')}")
            st.caption(m.get("createdAt", "")[:19].replace("T", " "))
    else:
        st.caption("No maintenance logs yet. Use the sidebar to log maintenance actions.")

# Loop control using rerun
time.sleep(sim_speed / 1000)
st.rerun()

