from detection.color_detector import HSVColorDetector

class YellowTapeDetector(HSVColorDetector):
    def __init__(self, min_area: int = 700):
    
        super().__init__(
            name="yellow_tape",
            hsv_ranges=[((20, 80, 80), (38, 255, 255))],
            min_area=min_area,
        )
