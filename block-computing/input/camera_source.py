import cv2


class CameraSource:
    """Live camera input source using OpenCV VideoCapture."""

    def __init__(
        self,
        camera_index: int = 0,
        resize_width: int | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        target_fps: float = 30.0,
    ):
        self.camera_index = camera_index
        self.resize_width = resize_width
        self.target_fps = target_fps

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}.")

        if frame_width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        if frame_height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        if target_fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, target_fps)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        if self.resize_width is not None:
            h, w = frame.shape[:2]
            if w != self.resize_width:
                scale = self.resize_width / float(w)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (self.resize_width, new_h))

        return True, frame

    def fps(self) -> float:
        detected_fps = self.cap.get(cv2.CAP_PROP_FPS)
        return detected_fps if detected_fps > 0 else self.target_fps

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
