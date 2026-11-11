import cv2
import numpy as np

from detection.base import Detection, ObstacleDetector


class PersonYoloDetector(ObstacleDetector):
    """
    YOLO-based person detector.

    Use model='yolov8n-seg.pt' for person masks/contours, which is much better for
    filtering a person out than motion detection boxes. Use 'yolov8n.pt' if you only
    want boxes. The first run may download the model weights via ultralytics.
    """

    name = "person_yolo"

    def __init__(
        self,
        model: str = "yolov8n-seg.pt",
        confidence: float = 0.35,
        imgsz: int = 640,
        mask_dilation: int = 15,
        max_det: int = 5,
        device: str | None = "auto",
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "PersonYoloDetector needs ultralytics. Install it with: pip install ultralytics"
            ) from exc

        self.model = YOLO(model)
        if device == "auto":
            try:
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        print(f"[yolo] running on {self.device}")
        self.confidence = confidence
        self.imgsz = imgsz
        self.mask_dilation = mask_dilation
        self.max_det = max_det
        self.kernel = np.ones((mask_dilation, mask_dilation), np.uint8) if mask_dilation > 0 else None

    def detect(self, frame, ignore_mask=None) -> list[Detection]:
        h, w = frame.shape[:2]

        results = self.model.predict(
            frame,
            classes=[0],  # COCO class 0 = person
            conf=self.confidence,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        masks = result.masks

        if boxes is None or len(boxes) == 0:
            return []

    
        best_idx = None
        best_score = -1.0

        for i, box in enumerate(boxes):
            conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 1.0

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy.tolist()
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            score = conf * area

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            return []

        box = boxes[best_idx]

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy.tolist()

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return []

        conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 1.0

        person_mask = np.zeros((h, w), dtype=np.uint8)
        contour = None

        if masks is not None and masks.data is not None and best_idx < len(masks.data):
            raw_mask = masks.data[best_idx].cpu().numpy()
            raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            person_mask[raw_mask > 0.5] = 255
        else:
            person_mask[y1:y2, x1:x2] = 255

        if self.kernel is not None:
            person_mask = cv2.dilate(person_mask, self.kernel, iterations=1)

        contours, _ = cv2.findContours(
            person_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(contour)
        else:
            x, y, bw, bh = x1, y1, x2 - x1, y2 - y1

        return [
            Detection(
                type=self.name,
                bbox=(x, y, bw, bh),
                contour=contour,
                mask=person_mask,
                confidence=conf,
            )
        ]
