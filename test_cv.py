import cv2
import numpy as np
print("OpenCV imported successfully.")
print(f"Version: {cv2.__version__}")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Camera 0 opened!")
    ret, frame = cap.read()
    if ret:
        print("Captured a frame!")
    else:
        print("Failed to capture frame.")
    cap.release()
else:
    print("Camera 0 failed.")
