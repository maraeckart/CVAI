"""Camera <-> projector calibration.

Projects 4 numbered markers onto the surface, finds them in the camera image and
computes a homography that maps camera-frame pixels -> projector pixels. main.py loads
it so balls simulated in camera space land on the real obstacles.

Usage (uses the same config.yaml as main.py, so input.mode must be "camera"):
    python calibrate_projector.py

Keys in the "Calibration - camera" window:
    a      auto-detect: projects each marker alone and finds it by image difference
    click  manual: click the markers in order 1, 2, 3, 4 (fallback if auto fails)
    r      reset points
    s      save (once 4 points are set)
    t      test: after saving, project a grid through the warp to check alignment
    q/ESC  quit
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from main import build_projector, build_source, load_config

CAM_WIN = "Calibration - camera"
MARGIN = 0.15  # marker distance from projector edge (fraction)
MARKER_RADIUS = 28


def marker_points(proj_w: int, proj_h: int) -> np.ndarray:
    mx, my = int(proj_w * MARGIN), int(proj_h * MARGIN)
    # order: 1 top-left, 2 top-right, 3 bottom-right, 4 bottom-left
    return np.array(
        [[mx, my], [proj_w - mx, my], [proj_w - mx, proj_h - my], [mx, proj_h - my]],
        dtype=np.float32,
    )


def pattern(proj_w, proj_h, pts, only=None):
    img = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)
    for i, (x, y) in enumerate(pts):
        if only is not None and i != only:
            continue
        cv2.circle(img, (int(x), int(y)), MARKER_RADIUS, (255, 255, 255), -1)
        if only is None:
            cv2.putText(img, str(i + 1), (int(x) + MARKER_RADIUS + 8, int(y) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    return img


def grab(source, settle_s=0.6):
    """Read frames for settle_s seconds (lets auto-exposure adapt), return a median-ish frame."""
    end = time.time() + settle_s
    frame = None
    while time.time() < end:
        ok, f = source.read()
        if ok:
            frame = f
        cv2.waitKey(1)
    frames = []
    for _ in range(3):
        ok, f = source.read()
        if ok:
            frames.append(f.astype(np.float32))
        cv2.waitKey(1)
    if frames:
        frame = np.median(np.stack(frames), axis=0).astype(np.uint8)
    return frame


def auto_detect(source, projector, pts):
    proj_w, proj_h = projector.width, projector.height
    projector.display(pattern(proj_w, proj_h, pts, only=-1), already_projector_space=True)
    base = cv2.cvtColor(grab(source, 1.0), cv2.COLOR_BGR2GRAY).astype(np.int16)

    found = []
    for i in range(len(pts)):
        projector.display(pattern(proj_w, proj_h, pts, only=i), already_projector_space=True)
        cur = cv2.cvtColor(grab(source), cv2.COLOR_BGR2GRAY).astype(np.int16)
        diff = np.clip(cur - base, 0, 255).astype(np.uint8)
        diff = cv2.GaussianBlur(diff, (7, 7), 0)
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if diff.max() < 25:
            print(f"[auto] marker {i + 1}: too faint (max diff {diff.max()}). Darken room / check camera sees projection.")
            return None
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            print(f"[auto] marker {i + 1}: not found")
            return None
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        found.append(centroids[best])
        print(f"[auto] marker {i + 1} at camera {centroids[best].round(1)}")

    projector.display(pattern(proj_w, proj_h, pts), already_projector_space=True)
    return [tuple(map(float, p)) for p in found]


def test_grid(proj_w, proj_h, cam_w, cam_h, H, step=80):
    """Grid drawn in camera space, warped to projector space."""
    cam = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
    for x in range(0, cam_w, step):
        cv2.line(cam, (x, 0), (x, cam_h), (0, 255, 0), 2)
    for y in range(0, cam_h, step):
        cv2.line(cam, (0, y), (cam_w, y), (0, 255, 0), 2)
    cv2.rectangle(cam, (2, 2), (cam_w - 3, cam_h - 3), (0, 0, 255), 4)
    return cv2.warpPerspective(cam, H, (proj_w, proj_h))


def main():
    config = load_config()
    if config.get("input", {}).get("mode") != "camera":
        print("WARNING: input.mode is not 'camera' in config.yaml - calibration needs the live camera.")

    proj_cfg = config.setdefault("output", {}).setdefault("projector", {})
    proj_cfg["enabled"] = True
    calib_path = Path(proj_cfg.get("calibration_file", "calibration/projector_homography.json"))

    source = build_source(config)
    projector = build_projector(config, use_calibration=False)
    proj_w, proj_h = projector.width, projector.height
    pts = marker_points(proj_w, proj_h)
    projector.display(pattern(proj_w, proj_h, pts), already_projector_space=True)

    clicked: list[tuple[float, float]] = []
    H = None
    testing = False

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append((float(x), float(y)))
            print(f"[click] marker {len(clicked)} at camera ({x}, {y})")

    cv2.namedWindow(CAM_WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(CAM_WIN, on_mouse)
    print(__doc__)

    cam_w = cam_h = None
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                print("Camera read failed.")
                break
            cam_h, cam_w = frame.shape[:2]
            view = frame.copy()
            for i, (x, y) in enumerate(clicked):
                cv2.circle(view, (int(x), int(y)), 8, (0, 0, 255), 2)
                cv2.putText(view, str(i + 1), (int(x) + 10, int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            if len(clicked) == 4:
                cv2.polylines(view, [np.array(clicked, dtype=np.int32)], True, (0, 255, 0), 2)
            status = f"points {len(clicked)}/4 | a=auto click=manual r=reset s=save t=test q=quit"
            cv2.putText(view, status, (10, cam_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(CAM_WIN, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                clicked.clear()
                testing = False
                projector.display(pattern(proj_w, proj_h, pts), already_projector_space=True)
            elif key == ord("a"):
                testing = False
                res = auto_detect(source, projector, pts)
                if res:
                    clicked[:] = res
                else:
                    print("[auto] failed - click the 4 markers manually instead.")
                    projector.display(pattern(proj_w, proj_h, pts), already_projector_space=True)
            elif key == ord("s"):
                if len(clicked) != 4:
                    print("Need 4 points before saving.")
                    continue
                H, _ = cv2.findHomography(np.array(clicked, dtype=np.float32), pts)
                if H is None:
                    print("Homography failed (points degenerate?). Reset and retry.")
                    continue
                calib_path.parent.mkdir(parents=True, exist_ok=True)
                with open(calib_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "homography": H.tolist(),
                            "camera_frame_size": [cam_w, cam_h],
                            "projector_size": [proj_w, proj_h],
                            "camera_points": clicked,
                            "projector_points": pts.tolist(),
                            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                        f,
                        indent=2,
                    )
                print(f"Saved calibration to {calib_path}. Press t to test, q to quit.")
            elif key == ord("t"):
                if H is None:
                    print("Save first (s).")
                    continue
                testing = not testing
                if testing:
                    projector.display(test_grid(proj_w, proj_h, cam_w, cam_h, H), already_projector_space=True)
                    print("Projected grid: in the camera view it should look straight and fill the frame.")
                else:
                    projector.display(pattern(proj_w, proj_h, pts), already_projector_space=True)
    finally:
        source.release()
        projector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
