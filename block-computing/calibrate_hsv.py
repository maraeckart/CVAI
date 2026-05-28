"""
Small helper for tuning HSV thresholds.

Usage:
    python calibrate_hsv.py videos/test.mp4

Click/inspect pixels by moving the mouse. The current HSV value is printed in the window.
"""

import sys

import cv2


current_hsv = None


def on_mouse(event, x, y, flags, data):
    global current_hsv
    frame, hsv = data
    if event == cv2.EVENT_MOUSEMOVE:
        current_hsv = hsv[y, x].tolist()


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/test.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read first frame.")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.namedWindow("HSV calibration")
    cv2.setMouseCallback("HSV calibration", on_mouse, [frame, hsv])

    while True:
        display = frame.copy()
        if current_hsv is not None:
            cv2.putText(
                display,
                f"HSV: {current_hsv}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow("HSV calibration", display)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
