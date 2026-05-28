import cv2


class CameraSource:
    """Live camera source. Useful later when moving from mockup video to real-time input."""

    def __init__(self, camera_index: int = 0, resize_width: int | None = None):
        self.camera_index = camera_index
        self.resize_width = resize_width
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}.")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        if self.resize_width:
            h, w = frame.shape[:2]
            if w != self.resize_width:
                scale = self.resize_width / float(w)
                frame = cv2.resize(frame, (self.resize_width, int(h * scale)))

        return True, frame

    def fps(self) -> float:
        return 30.0

    def release(self):
        self.cap.release()
