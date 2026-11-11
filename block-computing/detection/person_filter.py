import cv2
import numpy as np

from detection.person_motion import PersonMotionDetector


def build_person_detector(config: dict):
    cfg = config.get("person_filter", {})
    method = cfg.get("method", "yolo")

    if method == "none" or not cfg.get("enabled", True):
        return None

    if method == "motion":
        return PersonMotionDetector(min_area=cfg.get("min_area", 2500))

    if method == "yolo":
        from detection.person_yolo import PersonYoloDetector

        return PersonYoloDetector(
            model=cfg.get("model", "yolov8n-seg.pt"),
            confidence=float(cfg.get("confidence", 0.35)),
            imgsz=int(cfg.get("imgsz", 640)),
            mask_dilation=int(cfg.get("mask_dilation", 15)),
            max_det=int(cfg.get("max_det", 5)),
            device=cfg.get("device", "auto"),
        )

    raise ValueError(f"Unknown person_filter method: {method}")


def detections_to_mask(detections, shape):
    """Combine detection masks/contours/boxes into one 0/255 mask."""
    h, w = shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    for det in detections:
        if det.mask is not None:
            mask = det.mask
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            combined[mask > 0] = 255
        elif det.contour is not None:
            cv2.drawContours(combined, [det.contour], -1, 255, thickness=-1)
        else:
            x, y, bw, bh = det.bbox
            combined[y : y + bh, x : x + bw] = 255

    return combined
