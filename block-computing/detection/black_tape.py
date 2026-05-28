import cv2
import numpy as np

from detection.base import Detection, ObstacleDetector


class BlackTapeDetector(ObstacleDetector):
    def __init__(
        self,
        min_area: int = 700,
        max_area: int = 90000,
        value_threshold: int = 110,
        saturation_max: int = 180,
        min_pca_ratio: float = 4.0,
        close_kernel_size: int = 7,
        open_kernel_size: int = 2,
    ):
        self.name = "black_tape"
        self.min_area = min_area
        self.max_area = max_area
        self.value_threshold = value_threshold
        self.saturation_max = saturation_max
        self.min_pca_ratio = min_pca_ratio
        self.close_kernel_size = close_kernel_size
        self.open_kernel_size = open_kernel_size

    def _pca_ratio(self, contour):
        pts = contour.reshape(-1, 2).astype(np.float32)

        if len(pts) < 5:
            return 0.0

        _, _, eigenvalues = cv2.PCACompute2(pts, mean=None)

        major = float(eigenvalues[0, 0])
        minor = float(eigenvalues[1, 0])

        if minor < 1e-6:
            return 999.0

        return major / minor

    def detect(self, frame, ignore_mask=None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # Black tape = dark pixels, but avoid colorful dark shadows/noise.
        dark_mask = cv2.inRange(v, 0, self.value_threshold)
        low_sat_mask = cv2.inRange(s, 0, self.saturation_max)

        mask = cv2.bitwise_and(dark_mask, low_sat_mask)

        # Remove YOLO person area before detecting tape.
        if ignore_mask is not None:
            mask[ignore_mask > 0] = 0

        if self.open_kernel_size > 1:
            open_kernel = np.ones(
                (self.open_kernel_size, self.open_kernel_size),
                np.uint8,
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

        if self.close_kernel_size > 1:
            close_kernel = np.ones(
                (self.close_kernel_size, self.close_kernel_size),
                np.uint8,
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            if area > self.max_area:
                continue

            pca_ratio = self._pca_ratio(contour)

            # This replaces bbox aspect ratio.
            # It works better for diagonal or bent tape shapes.
            if pca_ratio < self.min_pca_ratio:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            object_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            cv2.drawContours(object_mask, [contour], -1, 255, thickness=-1)

            detections.append(
                Detection(
                    type=self.name,
                    bbox=(x, y, w, h),
                    contour=contour,
                    mask=object_mask,
                    confidence=1.0,
                )
            )

        return detections