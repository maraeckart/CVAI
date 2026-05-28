from detection.color_detector import HSVColorDetector


class PinkPostitDetector(HSVColorDetector):
    def __init__(
        self,
        min_area: int = 250,
        max_area: int | None = 20000,
        open_kernel_size: int = 3,
        close_kernel_size: int = 1,  # 1: nearby post-its do not get merged
    ):
    
        super().__init__(
            name="pink_postit",
            hsv_ranges=[
                ((145, 50, 80), (179, 255, 255)),
                ((0, 50, 100), (8, 255, 255)),
            ],
            min_area=min_area,
            max_area=max_area,
            open_kernel_size=open_kernel_size,
            close_kernel_size=close_kernel_size,
        )
