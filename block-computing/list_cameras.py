"""Print which OpenCV camera indices work and their resolution. Usage: python list_cameras.py"""
import cv2

for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            print(f"index {i}: {w}x{h}  -> saving preview to camera_{i}.jpg")
            cv2.imwrite(f"camera_{i}.jpg", frame)
        else:
            print(f"index {i}: opens but no frame")
    cap.release()
