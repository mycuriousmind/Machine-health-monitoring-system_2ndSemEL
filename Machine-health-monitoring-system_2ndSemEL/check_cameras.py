import cv2

def check_cameras():
    print("Checking available camera indices...")
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Index {i} is available.")
            cap.release()
        else:
            print(f"Index {i} is not available.")

if __name__ == "__main__":
    check_cameras()
