# OpenCV Real-Time Label Overlay

This project detects a predefined reference image (e.g., an Arduino board) using a webcam and overlays informative labels fixed to the object's perspective.

## Setup Instructions

1.  **Install Dependencies**:
    Make sure you have Python installed. Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Reference Image**:
    - The system uses `reference.png` as the target for detection.
    - An Arduino Uno reference image has been provided in the project folder.

3.  **Run the Detector**:
    Execute the Python script:
    ```bash
    python detector.py
    ```

## How it Works
- **ORB Feature Detection**: The script extracts unique keypoints from the reference image and the live webcam feed.
- **Feature Matching**: It matches features between the two images using Hamming distance.
- **Homography**: A perspective transformation matrix is calculated using RANSAC to map coordinates from the reference image onto the live video frame.
- **Relative Overlays**: Labels are defined using relative coordinates on the reference image and projected into the 3D space of the webcam feed.

## Shortcuts
- Press **'q'** to exit the application.
