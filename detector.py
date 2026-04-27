import cv2
import numpy as np

import random
import os
import json

# 0. Load ML Model for Fault Prediction
try:
    from tensorflow import keras
    model_path = "fault_detector_model.keras"
    if os.path.exists(model_path):
        ml_model = keras.models.load_model(model_path)
        print(f"ML Model loaded successfully from {model_path}")
    else:
        ml_model = None
        print("ML Model file not found. Fault prediction will be disabled.")
except ImportError:
    ml_model = None
    print("Tensorflow not found. Fault prediction will be disabled.")

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
    telemetry_cache = {lbl["name"]: 0 for lbl in labels}
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

                # 2. Stable Fixed HUD Telemetry (No moving labels)
                hud_x_start = w_frame - 300 # Moved left to ensure symbols show
                hud_y_start = 80
                
                for i, lbl in enumerate(labels):
                    # Only update values every 10 frames to stop flickering data
                    if frame_count % 10 == 0:
                        is_faulty = fault_triggers.get(lbl["name"], False)
                        
                        if "Signal" in lbl["name"]:
                            telemetry_cache[lbl["name"]] = random.randint(10, 30) if is_faulty else random.randint(85, 100)
                        elif "RPM" in lbl["name"]:
                            telemetry_cache[lbl["name"]] = random.randint(7000, 9000) if is_faulty else random.randint(2800, 3200)
                        elif "Vibration" in lbl["name"]:
                            if is_faulty:
                                axis_vibs = [random.uniform(5.0, 10.0) for _ in range(3)]
                            else:
                                axis_vibs = [random.uniform(-1.5, 1.5) for _ in range(3)]
                            
                            telemetry_cache[lbl["name"]] = sum(axis_vibs) / 3.0
                            vibration_buffer = np.roll(vibration_buffer, -1, axis=0)
                            vibration_buffer[-1] = axis_vibs
                            
                            if ml_model is not None:
                                x_input = vibration_buffer[np.newaxis, ...]
                                prob = ml_model.predict(x_input, verbose=0)[0][0]
                                if prob > 0.5:
                                    health_status = "FAULTY"
                                    health_color = (0, 0, 255)
                                else:
                                    health_status = "HEALTHY"
                                    health_color = (0, 255, 0)
                        else: # Temperatures
                            telemetry_cache[lbl["name"]] = random.uniform(85.0, 110.0) if is_faulty else random.uniform(34.0, 37.0)


                    
                    # Flashing Effect if Faulty
                    is_faulty = fault_triggers.get(lbl["name"], False)
                    show_item = True
                    if is_faulty and (frame_count // 5) % 2 == 0:
                        show_item = False
                    
                    if show_item:
                        val = telemetry_cache[lbl["name"]]
                        val_str = f"{val:.1f}" if isinstance(val, float) else str(val)
                        full_label = f"{lbl['name']}: {val_str}{lbl['unit']}"
                        
                        color = (0, 0, 255) if is_faulty else lbl["color"]
                        
                        # Fixed HUD Position
                        tx, ty = hud_x_start, hud_y_start + (i * 45)
                        
                        (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        bg_x1, bg_y1 = tx, ty - th - 10
                        bg_x2, bg_y2 = tx + tw + 20, ty + 5
                        
                        # Add Alert Symbol if Faulty
                        if is_faulty:
                            # Draw Warning Triangle
                            tri_pts = np.array([
                                [tx - 35, ty + 5], # Bottom left
                                [tx - 5, ty + 5],  # Bottom right
                                [tx - 20, ty - 25] # Top middle
                            ], np.int32)
                            cv2.drawContours(display_frame, [tri_pts], 0, (0, 0, 255), -1)
                            cv2.putText(display_frame, "!", (tx - 24, ty - 2), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        cv2.rectangle(display_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
                        cv2.rectangle(display_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
                        cv2.putText(display_frame, full_label, (tx + 10, ty - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


                # -- Dashboard Sync (Optimized: Outside loop) --
                if frame_count % 10 == 0:
                    try:
                        # Save current telemetry
                        shared_payload = {
                            "telemetry": telemetry_cache,
                            "fault_triggers": fault_triggers,
                            "health": {"status": health_status, "prob": float(prob) if 'prob' in locals() else 0.0}
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
                hud_x, hud_y = 20, h_frame - 60
                cv2.rectangle(display_frame, (hud_x, hud_y), (hud_x + 200, hud_y + 40), (20, 20, 20), -1)
                cv2.rectangle(display_frame, (hud_x, hud_y), (hud_x + 200, hud_y + 40), health_color, 2)
                cv2.putText(display_frame, f"HEALTH: {health_status}", (hud_x + 10, hud_y + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, health_color, 2)

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
