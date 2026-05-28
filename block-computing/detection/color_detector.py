import cv2
import numpy as np

from detection.base import Detection, ObstacleDetector


class HSVColorDetector(ObstacleDetector):
    """Generic HSV color detector with contour extraction."""

    def __init__(
        self,
        name: str,
        hsv_ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
        min_area: int = 500,
        max_area: int | None = None,
        open_kernel_size: int = 3,
        close_kernel_size: int = 1,
    ):
        self.name = name
        self.hsv_ranges = hsv_ranges
        self.min_area = min_area
        self.max_area = max_area
        self.open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8) if open_kernel_size > 1 else None
        self.close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8) if close_kernel_size > 1 else None

    def make_mask(self, frame, ignore_mask=None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        final_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for lower, upper in self.hsv_ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_np, upper_np)
            final_mask = cv2.bitwise_or(final_mask, mask)

        if ignore_mask is not None:
            final_mask[ignore_mask > 0] = 0

        if self.open_kernel is not None:
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, self.open_kernel)
        if self.close_kernel is not None:
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, self.close_kernel)
        return final_mask

    def detect(self, frame, ignore_mask=None) -> list[Detection]:
        mask = self.make_mask(frame, ignore_mask=ignore_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue

            # Per-object mask. Important: this lets collision use the actual contour,
            # not the full-frame merged color mask.
            object_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            cv2.drawContours(object_mask, [contour], -1, 255, thickness=-1)

            x, y, w, h = cv2.boundingRect(contour)
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
