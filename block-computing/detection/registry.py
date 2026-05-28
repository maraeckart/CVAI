from detection.black_tape import BlackTapeDetector
from detection.yellow_tape import YellowTapeDetector
from detection.pink_postit import PinkPostitDetector



def build_detectors(config: dict):
    """
    Build only the non-YOLO colored obstacle detectors.
    """
    detector_cfg = config.get("detectors", {})
    detectors = []

    if detector_cfg.get("black_tape", {}).get("enabled", False):
        cfg = detector_cfg["black_tape"]
        detectors.append(
            BlackTapeDetector(
                min_area=cfg.get("min_area", 700),
                max_area=cfg.get("max_area", 90000),
                value_threshold=cfg.get("value_threshold", 110),
                saturation_max=cfg.get("saturation_max", 180),
                min_pca_ratio=cfg.get("min_pca_ratio", 4.0),
                close_kernel_size=cfg.get("close_kernel_size", 7),
                open_kernel_size=cfg.get("open_kernel_size", 2),
            )
        )

    if detector_cfg.get("yellow_tape", {}).get("enabled", False):
        cfg = detector_cfg["yellow_tape"]
        detectors.append(
            YellowTapeDetector(
                min_area=cfg.get("min_area", 700),
            )
        )

    if detector_cfg.get("pink_postit", {}).get("enabled", False):
        cfg = detector_cfg["pink_postit"]
        detectors.append(
            PinkPostitDetector(
                min_area=cfg.get("min_area", 250),
                max_area=cfg.get("max_area", 20000),
                open_kernel_size=cfg.get("open_kernel_size", 3),
                close_kernel_size=cfg.get("close_kernel_size", 1),
            )
        )

    return detectors
