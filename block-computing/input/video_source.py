import cv2


class VideoSource:
    """Frame source for prerecorded video files."""

    def __init__(self, path: str, resize_width: int | None = None):
        self.path = path
        self.resize_width = resize_width
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(
                f"Could not open video: {path}. Put a video at this path or update config.yaml."
            )

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
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 1 else 30.0

    def release(self):
        self.cap.release()
