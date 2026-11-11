import cv2
import numpy as np

from detection.base import Detection, ObstacleDetector


class PersonMotionDetector(ObstacleDetector):
    name = "person_motion"

    def __init__(self, min_area: int = 2500):
        self.min_area = min_area
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
        self.kernel = np.ones((5, 5), np.uint8)

    def detect(self, frame, ignore_mask=None) -> list[Detection]:
        mask = self.bg_sub.apply(frame)
        if ignore_mask is not None:
            mask[ignore_mask > 0] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(
                Detection(
                    type=self.name,
                    bbox=(x, y, w, h),
                    contour=contour,
                    mask=mask,
                    confidence=1.0,
                )
            )
        return detections
