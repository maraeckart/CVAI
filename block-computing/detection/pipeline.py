import cv2
import numpy as np

from detection.person_filter import build_person_detector, detections_to_mask
from detection.registry import build_detectors


class DetectionPipeline:
    """
    detection pipeline.

    Currently:
    - YOLO is used for person segmentation/filtering/occlusion.
    - HSV detectors are used for colored obstacles such as post-its and tape.

    Later:
    - Replace the HSV detectors with a custom YOLO segmentation model that outputs
      pink_postit, yellow_tape, black_tape, etc. The rest of the app can stay the same.
    """

    def __init__(self, config: dict):
        self.config = config
        self.person_detector = build_person_detector(config)
        self.obstacle_detectors = build_detectors(config)
        self.rules = config.get("object_rules", {})

    @property
    def enabled_names(self):
        names = []
        if self.person_detector is not None:
            names.append(self.person_detector.name)
        names.extend(detector.name for detector in self.obstacle_detectors)
        return names

    def _rule(self, object_type: str) -> dict:
        return self.rules.get(object_type, {})

    def _combined_mask(self, detections, frame_shape):
        if not detections:
            h, w = frame_shape[:2]
            return np.zeros((h, w), dtype=np.uint8)
        return detections_to_mask(detections, frame_shape)

    def detect(self, frame):
        """
        Returns:
            obstacles: detections that should collide with balls
            filter_detections: detections used only for filtering/debug drawing
            filter_mask: mask used to remove pixels before color obstacle detection
            occlusion_mask: mask used to hide balls behind real foreground objects
        """
        all_filter_detections = []
        obstacles = []
        occlusion_masks = []

        h, w = frame.shape[:2]

        #  YOLO person pass: always first, so the person can be removed from
        #  color obstacle detection and can hide dots visually.
        person_detections = []
        if self.person_detector is not None:
            person_detections = self.person_detector.detect(frame)

        for det in person_detections:
            rule = self._rule(det.type)

            if rule.get("use_as_filter", True):
                det.is_filter_only = True
                all_filter_detections.append(det)

            if rule.get("use_as_occluder", True) and det.mask is not None:
                occlusion_masks.append(det.mask)

            if rule.get("use_as_obstacle", False):
                det.is_filter_only = False
                obstacles.append(det)

        filter_mask = self._combined_mask(all_filter_detections, frame.shape)

        # bottom crop in case to HSV/color detectors ignore the floor 
        crop_ratio = float(
            self.config.get("processing", {}).get("detection_crop_bottom_ratio", 1.0)
        )
        crop_ratio = max(0.0, min(1.0, crop_ratio))

        crop_y = int(h * crop_ratio)

        if crop_y < h:
            filter_mask[crop_y:h, :] = 255

        # HSV/color obstacle pass
        for detector in self.obstacle_detectors:
            detections = detector.detect(frame, ignore_mask=filter_mask)

            for det in detections:
                rule = self._rule(det.type)

                if rule.get("use_as_obstacle", True):
                    obstacles.append(det)

                if rule.get("use_as_filter", False):
                    det.is_filter_only = True
                    all_filter_detections.append(det)

                if rule.get("use_as_occluder", False) and det.mask is not None:
                    occlusion_masks.append(det.mask)

        # combination of all occluder masks (mainly only the YOLO person mask)
        occlusion_mask = np.zeros((h, w), dtype=np.uint8)

        for mask in occlusion_masks:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            occlusion_mask[mask > 0] = 255

        return obstacles, all_filter_detections, filter_mask, occlusion_mask