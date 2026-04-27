import cv2
import numpy as np

import random

def main():
    # 1. Setup ORB Detector - More features for better accuracy
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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
        {"name": "Battery", "pos": (0.5, 0.8), "unit": "%", "color": (255, 0, 255)}    # Magenta
    ]

    # Telemetry Data Cache (to prevent flickering values)
    telemetry_cache = {lbl["name"]: 0 for lbl in labels}
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
    
    # Smoothing & Persistence - Reduced for instant disappearance
    prev_M = None
    frames_since_detect = 0
    MAX_LOST_FRAMES = 2 # Disappear quickly if lost
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

                    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                    
                    if M is not None:
                        h_r, w_r = img_ref.shape[:2]
                        p = np.float32([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]]).reshape(-1, 1, 2)
                        d = cv2.perspectiveTransform(p, M)
                        
                        area = cv2.contourArea(d)
                        ref_area = w_r * h_r
                        
                        # Accept if area is roughly in range
                        if ref_area * 0.05 < area < ref_area * 20: # Loosened for distance
                            if prev_M is not None:
                                current_M = prev_M * 0.5 + M * 0.5
                            else:
                                current_M = M
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
                # 1. Bounding Box
                h_ref, w_ref = img_ref.shape[:2]
                pts = np.float32([[0, 0], [0, h_ref-1], [w_ref-1, h_ref-1], [w_ref-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, current_M)
                cv2.polylines(display_frame, [np.int32(dst)], True, (255, 255, 255), 1, cv2.LINE_AA)

                # 2. Dynamic Telemetry Labels
                box_center_x = (dst[0][0][0] + dst[2][0][0]) / 2
                box_center_y = (dst[0][0][1] + dst[2][0][1]) / 2

                for i, lbl in enumerate(labels):
                    pt_original = label_pts[i].reshape(-1, 1, 2)
                    pt_transformed = cv2.perspectiveTransform(pt_original, current_M)
                    px, py = int(pt_transformed[0][0][0]), int(pt_transformed[0][0][1])

                    if 0 <= px < w_frame and 0 <= py < h_frame:
                        # Only update values every 10 frames to stop flickering data
                        if frame_count % 10 == 0:
                            if "Signal" in lbl["name"] or "Battery" in lbl["name"]:
                                telemetry_cache[lbl["name"]] = random.randint(85, 100)
                            else:
                                telemetry_cache[lbl["name"]] = random.uniform(34.0, 37.0)
                        
                        val = telemetry_cache[lbl["name"]]
                        val_str = f"{val:.1f}" if isinstance(val, float) else str(val)
                        full_label = f"{lbl['name']}: {val_str}{lbl['unit']}"

                        # Offset Label
                        dx, dy = (70 if px > box_center_x else -70), -50 - (i*10)
                        tx, ty = px + dx, py + dy

                        # Draw HUD Elements with parameter color
                        color = lbl["color"]
                        cv2.line(display_frame, (px, py), (tx, ty), color, 1, cv2.LINE_AA)
                        cv2.circle(display_frame, (px, py), 4, color, -1)
                        
                        (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        bg_x1, bg_y1 = (tx, ty - th - 10) if dx > 0 else (tx - tw - 10, ty - th - 10)
                        bg_x2, bg_y2 = (tx + tw + 10, ty) if dx > 0 else (tx, ty)
                        
                        cv2.rectangle(display_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
                        cv2.rectangle(display_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1) # Border
                        cv2.putText(display_frame, full_label, (bg_x1 + 5, bg_y2 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            cv2.putText(display_frame, "STABLE HUD ACTIVE - 'R' TO RESET", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('AR Stable HUD', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s') and not is_detecting:
            # 1. Capture ROI
            roi_capture = frame[y1:y2, x1:x2]
            temp_gray = cv2.cvtColor(roi_capture, cv2.COLOR_BGR2GRAY)
            
            # 2. Find points to determine exact object size
            temp_kp, _ = orb.detectAndCompute(temp_gray, None)
            
            if len(temp_kp) > 20:
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
