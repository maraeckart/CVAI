import json
from pathlib import Path

import cv2
import numpy as np


def load_calibration(path: str | Path):
    """Load a camera->projector homography written by calibrate_projector.py.

    Returns (H, meta) or (None, None) if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["homography"], dtype=np.float64), data


class ProjectorStub:
    """Fullscreen projector window on a secondary display with camera->projector warping.

    Frames passed to display() are in camera-frame coordinates (after resize_width).
    If a homography is set, they are warped into projector pixel space so that a ball
    drawn at camera pixel (x, y) lands on the physical spot the camera sees at (x, y).
    """

    def __init__(
        self,
        window_name: str = "Projector Output",
        width: int = 1920,
        height: int = 1080,
        fullscreen: bool = True,
        screen_x_offset: int = 1920,
        screen_y_offset: int = 0,
        warp_matrix: np.ndarray | None = None,
    ):
        self.window_name = window_name
        self.width = int(width)
        self.height = int(height)
        self.fullscreen = fullscreen
        self.screen_x_offset = screen_x_offset
        self.screen_y_offset = screen_y_offset
        self.warp_matrix = warp_matrix

        self._init_window()

    def _init_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Show something first: on macOS a window that never got an image may ignore
        # move/fullscreen requests.
        cv2.imshow(self.window_name, np.zeros((self.height, self.width, 3), dtype=np.uint8))
        cv2.waitKey(1)

        # Move onto the beamer display before going fullscreen, so fullscreen
        # happens on that screen and not on the laptop screen.
        cv2.moveWindow(self.window_name, self.screen_x_offset, self.screen_y_offset)
        cv2.waitKey(1)

        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(self.window_name, self.width, self.height)
        cv2.waitKey(1)

    def to_projector(self, frame: np.ndarray) -> np.ndarray:
        """Map a camera-space frame into projector pixel space."""
        if self.warp_matrix is not None:
            return cv2.warpPerspective(
                frame,
                self.warp_matrix,
                (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            return cv2.resize(frame, (self.width, self.height))
        return frame

    def display(self, frame: np.ndarray, already_projector_space: bool = False) -> np.ndarray:
        output_frame = frame if already_projector_space else self.to_projector(frame)
        cv2.imshow(self.window_name, output_frame)
        return output_frame

    def set_warp_matrix(self, matrix: np.ndarray | None):
        self.warp_matrix = matrix

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
