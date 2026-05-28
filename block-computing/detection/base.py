from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Detection:
    type: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    contour: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    confidence: float = 1.0
    is_filter_only: bool = False

    @property
    def rect_xyxy(self):
        x, y, w, h = self.bbox
        return x, y, x + w, y + h


class ObstacleDetector:
    name = "base"

    def detect(self, frame, ignore_mask=None) -> list[Detection]:
        """Return detections. ignore_mask is a 0/255 mask for pixels to ignore."""
        raise NotImplementedError
