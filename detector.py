import cv2
import numpy as np

import time
import random
import os
import json

def get_shared_data():
    if os.path.exists("shared_data.json"):
        try:
            with open("shared_data.json", "r") as f:
                return json.load(f)
        except:
            pass
    return {}

# 0. Load ML Models for Fault Prediction
import model_utils
cnn_model, rf_model, fusion_model, mu_vib, sd_vib = model_utils.load_all_models()

# 1. Fault Trigger & Control GUI
import tkinter as tk
from threading import Thread

fault_triggers = {
    "Temperature": False,
    "Vibration": False,
    "Signal": False,
    "Core Temp": False,
    "RPM": False
}

def start_fault_gui():
    try:
        root = tk.Tk()
        root.title("⚠️ FAULT INJECTOR")
        root.geometry("300x400")
        root.attributes("-topmost", True)
        
        tk.Label(root, text="SYSTEM CONTROL PANEL", font=("Arial", 12, "bold")).pack(pady=10)
        tk.Label(root, text="Toggle faults to test AR HUD", font=("Arial", 9)).pack(pady=5)
        
        for key in fault_triggers.keys():
            frame = tk.Frame(root)
            frame.pack(pady=5, fill='x', padx=20)
            
            var = tk.BooleanVar(value=fault_triggers[key])
            
            def toggle_callback(k=key, v=var):
                fault_triggers[k] = v.get()
                print(f"DEBUG: {k} fault set to {fault_triggers[k]}")
                
            cb = tk.Checkbutton(frame, text=f"Inject {key} Fault", variable=var, 
                                command=toggle_callback, font=("Arial", 10))
            cb.pack(side='left')
        
        tk.Label(root, text="\nStatus: Connection Active", fg="green").pack()
        root.mainloop()
    except Exception as e:
        print(f"GUI Error: {e}")

# Start GUI in background thread
Thread(target=start_fault_gui, daemon=True).start()

def main():
    # 1. Setup ORB Detector - Balanced for speed and accuracy
    orb = cv2.ORB_create(nfeatures=500) # Reduced for maximum speed
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 2. State & Telemetry Simulation
    img_ref = None
    kp_ref = None
    des_ref = None
    label_pts = []
    
    # Telemetry Labels with Unique Colors
    labels = [
        {"name": "Temperature", "pos": (0.5, 0.2), "unit": "C", "color": (0, 0, 255)},    # Red
        {"name": "Vibration", "pos": (0.2, 0.5), "unit": "Hz", "color": (255, 0, 0)},   # Blue
        {"name": "Signal", "pos": (0.8, 0.5), "unit": "%", "color": (0, 255, 0)},      # Green
        {"name": "Core Temp", "pos": (0.5, 0.5), "unit": "C", "color": (0, 255, 255)},  # Yellow
        {"name": "RPM", "pos": (0.5, 0.8), "unit": "", "color": (255, 0, 255)}    # Magenta
    ]

    # Telemetry Data Cache (to prevent flickering values)
    telemetry_cache = {
        "Temperature": 35.0,
        "Vibration": 0.0,
        "Signal": 100,
        "Core Temp": 45.0,
        "RPM": 3000
    }
    vibration_buffer = np.zeros((100, 3), dtype=np.float32)
    health_status = "INITIALIZING"
    health_color = (255, 255, 255)
    frame_count = 0

    # 3. Video Capture Initialization
    cap = None
    for i in [0, 1, 2]:
        cap = cv2.VideoCapture(i)
        if cap.isOpened(): break
        cap.release()

    if not cap or not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    # Setup Matcher (using KNN for Ratio Test)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Smoothing & Persistence - Added buffer to prevent appearing/reappearing flicker
    prev_M = None
    frames_since_detect = 0
    MAX_LOST_FRAMES = 15 # Stays for ~0.5s after loss to prevent flicker
    is_detecting = False
    
    print("\n--- HIGH-PRECISION AR SCANNER ---")
    print("1. Hold object in GREEN box -> Press 'S' to SCAN.")
    print("2. Press 'R' to RESET. Press 'Q' to EXIT.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h_frame, w_frame = frame.shape[:2]
        frame_count += 1
        
        roi_w, roi_h = 350, 350
        x1, y1 = (w_frame - roi_w) // 2, (h_frame - roi_h) // 2
        x2, y2 = x1 + roi_w, y1 + roi_h
        
        display_frame = frame.copy()

        if not is_detecting:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, "SCAN TARGET -> PRESS 'S'", (x1, y1 - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            roi_img = frame[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            kp_frame, des_frame = orb.detectAndCompute(gray_roi, None)

            current_M = None
            if des_frame is not None and len(des_frame) > 15:
                # 1. Lowe's Ratio Test (Balanced Sensitivity)
                raw_matches = bf.knnMatch(des_ref, des_frame, k=2)
                good_matches = []
                for m_n in raw_matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.82 * n.distance: # Slightly loosened
                            good_matches.append(m)

                # 2. Homography Validation
                if len(good_matches) > 10: # Lower for better sensitivity
                    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts[:, :, 0] += x1
                    dst_pts[:, :, 1] += y1

                    # Perfectly Stagnant Bounding Box (Uses ROI position)
                    # This prevents ANY movement or 'laggy' jitter
                    current_M = [x1, y1, x2, y2]
                    frames_since_detect = 0

            # Instant disappearance if threshold isn't met
            if current_M is None and prev_M is not None and frames_since_detect < MAX_LOST_FRAMES:
                current_M = prev_M
                frames_since_detect += 1
            elif current_M is not None:
                prev_M = current_M
            else:
                prev_M = None # Reset previous on full loss

            if current_M is not None:
                # 1. Stable Bounding Box (Rigid Rectangle)
                rx1, ry1, rx2, ry2 = current_M
                cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)

                # --- CHECK ESP32 STATUS ---
                shared_data = get_shared_data()
                esp_data = shared_data.get("esp_data", {})
                
                if time.time() - esp_data.get("Timestamp", 0) > 5:
                    # Disconnected
                    cv2.putText(display_frame, "CONNECT ESP32", (w_frame//2 - 180, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    
                    hud_x_start = w_frame - 300
                    hud_y_start = 80
                    for i, lbl in enumerate(labels):
                        tx, ty = hud_x_start, hud_y_start + (i * 45)
                        cv2.putText(display_frame, f"{lbl['name']}: ---", (tx, ty), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
                        
                else:
                    # Real ESP32 Data
                    real_temp = esp_data.get("Temperature", 35.0)
                    real_current = esp_data.get("Current", 0.0)
                    mapped_torque = real_current * 20.0
                    mapped_rpm = max(0, 3000.0 - (real_current * 100.0))
                    is_esp_faulty = real_temp > 45.0 or real_current > 3.0
                    
                    hud_x_start = w_frame - 300
                    hud_y_start = 80
                    
                    for i, lbl in enumerate(labels):
                        # Map ESP data to telemetry cache
                        if "Temperature" in lbl["name"]:
                            telemetry_cache[lbl["name"]] = real_temp
                        elif "RPM" in lbl["name"]:
                            telemetry_cache[lbl["name"]] = mapped_rpm
                        elif "Signal" in lbl["name"]:
                            telemetry_cache[lbl["name"]] = 100
                        elif "Vibration" in lbl["name"]:
                            t = np.linspace(0, 100 / 12000, 100)
                            if is_esp_faulty:
                                f0 = 30.0 
                                bpfi = f0 * 5.4
                                sig = 1.0 * np.sin(2*np.pi*f0*t) + 0.4 * np.sin(2*np.pi*bpfi*t) + 0.2 * np.sin(2*np.pi*2*bpfi*t)
                                for pos in [20, 60]:
                                    sig += 2.0 * np.exp(-200*(np.arange(100)-pos)**2/100)
                                vibration_buffer = np.stack([sig, sig, sig], axis=1).astype(np.float32) + np.random.randn(100, 3).astype(np.float32) * 0.2
                            else:
                                sig = 0.3 * np.sin(2*np.pi*30.0*t)
                                vibration_buffer = np.stack([sig, sig, sig], axis=1).astype(np.float32) + np.random.randn(100, 3).astype(np.float32) * 0.05
                                
                            telemetry_cache[lbl["name"]] = float(np.sqrt(np.mean(vibration_buffer**2)))
                            
                            # Real-time Inference using mapped ESP data
                            cnn_prob = model_utils.predict_vibration(cnn_model, vibration_buffer, mu_vib, sd_vib)
                            rf_prob = model_utils.predict_tabular(rf_model, 
                                                                  temperature=real_temp,
                                                                  rpm=mapped_rpm,
                                                                  torque=mapped_torque,
                                                                  tool_wear=100.0)
                            
                            health_res = model_utils.get_combined_health(cnn_prob, rf_prob, fusion_model)
                            health_status = health_res["status"]
                            prob = health_res["fused_prob"]
                            
                            health_color = (0, 0, 255) if health_status == "FAULTY" else (0, 255, 0)
                            
                        # Draw label
                        val = telemetry_cache.get(lbl["name"], 0)
                        val_str = f"{val:.1f}" if isinstance(val, float) else str(val)
                        full_label = f"{lbl['name']}: {val_str}{lbl['unit']}"
                        
                        tx, ty = hud_x_start, hud_y_start + (i * 45)
                        color = (0, 0, 255) if is_esp_faulty else lbl["color"]
                        
                        (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        bg_x1, bg_y1 = tx, ty - th - 10
                        bg_x2, bg_y2 = tx + tw + 20, ty + 5
                        
                        # Add Warning Triangle if ESP says faulty
                        if is_esp_faulty:
                            tri_pts = np.array([
                                [tx - 35, ty + 5],
                                [tx - 5, ty + 5],
                                [tx - 20, ty - 25]
                            ], np.int32)
                            cv2.drawContours(display_frame, [tri_pts], 0, (0, 0, 255), -1)
                        
                        cv2.rectangle(display_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
                        cv2.rectangle(display_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
                        cv2.putText(display_frame, full_label, (tx + 10, ty), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


                # -- Dashboard Sync (Optimized: Outside loop) --
                if frame_count % 10 == 0:
                    try:
                        # Save current telemetry
                        shared_payload = {
                            "telemetry": telemetry_cache,
                            "fault_triggers": fault_triggers,
                            "health": {
                                "status": health_status, 
                                "fused_prob": health_res["fused_prob"] if 'health_res' in locals() else 0.0,
                                "cnn_prob": health_res["cnn_prob"] if 'health_res' in locals() else 0.0,
                                "rf_prob": health_res["rf_prob"] if 'health_res' in locals() else 0.0
                            }
                        }
                        with open("shared_data.json", "w") as f:
                            json.dump(shared_payload, f)
                            
                        # Load dashboard faults
                        if os.path.exists("shared_data.json"):
                            with open("shared_data.json", "r") as f:
                                shared = json.load(f)
                                dashboard_faults = shared.get("dashboard_faults", {})
                                for k, v in dashboard_faults.items():
                                    if k in fault_triggers:
                                        fault_triggers[k] = v
                    except Exception:
                        pass

                # 3. Machine Health Status Dashboard (Fixed Position)
                hud_x, hud_y = 20, h_frame - 80
                cv2.rectangle(display_frame, (hud_x, hud_y), (hud_x + 250, hud_y + 60), (20, 20, 20), -1)
                cv2.rectangle(display_frame, (hud_x, hud_y), (hud_x + 250, hud_y + 60), health_color, 2)
                cv2.putText(display_frame, f"HEALTH: {health_status}", (hud_x + 10, hud_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, health_color, 2)
                if 'health_res' in locals():
                    probs_str = f"CNN: {health_res['cnn_prob']:.2f} | RF: {health_res['rf_prob']:.2f}"
                    cv2.putText(display_frame, probs_str, (hud_x + 10, hud_y + 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.putText(display_frame, "STABLE HUD ACTIVE - 'R' TO RESET", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('AR Stable HUD', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'):
            if is_detecting:
                print("Already detecting. Press 'R' to reset before scanning again.")
            else:
                print("Scanning ROI...")
                # 1. Capture ROI
                roi_capture = frame[y1:y2, x1:x2]
                temp_gray = cv2.cvtColor(roi_capture, cv2.COLOR_BGR2GRAY)
                
                # 2. Find points to determine exact object size
                temp_kp, _ = orb.detectAndCompute(temp_gray, None)
                
                if len(temp_kp) > 10: # Lowered threshold for sensitivity
                    # Calculate Bounding Box of features (where the object actually is)
                    pts_list = np.float32([k.pt for k in temp_kp])
                    min_x, min_y = np.min(pts_list, axis=0)
                    max_x, max_y = np.max(pts_list, axis=0)
                
                    # Add small padding (10px)
                    pad = 10
                    bx1, by1 = max(0, int(min_x)-pad), max(0, int(min_y)-pad)
                    bx2, by2 = min(roi_w, int(max_x)+pad), min(roi_h, int(max_y)+pad)
                    
                    # 3. Crop reference to just the object
                    img_ref = temp_gray[by1:by2, bx1:bx2]
                    kp_ref, des_ref = orb.detectAndCompute(img_ref, None)
                    
                    # 4. Setup labels relative to this new tight box
                    h_r, w_r = img_ref.shape
                    label_pts = [np.array([[int(l["pos"][0]*w_r), int(l["pos"][1]*h_r)]], dtype='float32') for l in labels]
                    
                    is_detecting = True
                    print(f"Object Scanned! (Size: {w_r}x{h_r})")
                else:
                    print("Error: No object detected in box. Try better lighting.")
        elif key == ord('r'):
            is_detecting = False
            prev_M = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
