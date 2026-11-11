import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from detection.pipeline import DetectionPipeline
from input.camera_source import CameraSource
from input.video_source import VideoSource
from physics.ball import create_balls_from_config
from physics.collision import resolve_ball_mask_collision
from physics.particle_collusion import resolve_ball_ball_collisions
from physics.spawner import BallSpawner
from rendering.overlay import OverlayRenderer
from rendering.projector_stub import ProjectorStub, load_calibration


CONFIG_PATH = "config.yaml"


def load_config(path: str = CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_source(config: dict):
    input_cfg = config.get("input", {})
    processing_cfg = config.get("processing", {})
    resize_width = processing_cfg.get("resize_width")

    mode = input_cfg.get("mode", "video")
    if mode == "video":
        return VideoSource(input_cfg.get("path", "videos/test.mp4"), resize_width=resize_width)
    if mode == "camera":
        return CameraSource(
            camera_index=input_cfg.get("camera_index", 0),
            resize_width=resize_width,
            frame_width=input_cfg.get("capture_width"),
            frame_height=input_cfg.get("capture_height"),
            target_fps=input_cfg.get("capture_fps", 30.0),
        )
    raise ValueError(f"Unknown input mode: {mode}")


def offset_matrix(dx: float, dy: float) -> np.ndarray:
    """Translation in projector pixels, applied after the camera->projector homography."""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def build_projector(config: dict, use_calibration: bool = True):
    proj_cfg = config.get("output", {}).get("projector", {})
    if not proj_cfg.get("enabled", False):
        return None

    warp, meta = None, None
    if use_calibration:
        calib_path = proj_cfg.get("calibration_file", "calibration/projector_homography.json")
        warp, meta = load_calibration(calib_path)
        if warp is None:
            print(f"[projector] No calibration at {calib_path} -> output is just stretched to the beamer.")
            print("[projector] Run: python calibrate_projector.py")
        else:
            print(f"[projector] Loaded calibration {calib_path} (camera frame {meta.get('camera_frame_size')})")

    projector = ProjectorStub(
        width=proj_cfg.get("width", 1920),
        height=proj_cfg.get("height", 1080),
        fullscreen=proj_cfg.get("fullscreen", True),
        screen_x_offset=proj_cfg.get("screen_x_offset", 1920),
        screen_y_offset=proj_cfg.get("screen_y_offset", 0),
        warp_matrix=warp,
    )
    projector.content = proj_cfg.get("content", "balls")  # "balls" (black bg) or "debug" (full overlay)
    projector.calib_meta = meta
    return projector


def main():
    config = load_config()

    output_cfg = config.get("output", {})
    save_video = bool(output_cfg.get("save_video", False))
    save_path = output_cfg.get("path", "output/result.mp4") if save_video else None
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    source = build_source(config)
    detection_pipeline = DetectionPipeline(config)
    renderer = OverlayRenderer(save_path=save_path, fps=source.fps(), config=config)

    projector = build_projector(config)
    calib_size_checked = False

    # Rectified mode: warp each camera frame into the projected area and run
    # detection + physics there. Then the simulation world == what is on the wall:
    # balls spawn at the top of the projection, pile on its bottom edge and bounce
    # off its sides. The projector output then needs no extra warp, only a resize.
    # Fine-tune offset (projector pixels) on top of the calibration. Nudge it live with
    # j/l (left/right) and i/k (up/down) in a program window, then copy the printed
    # values into config.yaml.
    rectify_H, rectify_size = None, None
    proj_cfg = output_cfg.get("projector", {})
    offset = [float(proj_cfg.get("offset_x", 0)), float(proj_cfg.get("offset_y", 0))]
    nudge_step = float(proj_cfg.get("nudge_step", 10))
    base_H = projector.warp_matrix if projector else None
    rectify = bool(base_H is not None and proj_cfg.get("simulate_in_projection", True))
    scale = None
    if rectify:
        sim_w = int(proj_cfg.get("sim_width", 960))
        sim_h = int(round(sim_w * projector.height / projector.width))
        scale = np.diag([sim_w / projector.width, sim_h / projector.height, 1.0])
        rectify_size = (sim_w, sim_h)
        print(f"[projector] simulating inside the projected area ({sim_w}x{sim_h})")

    def apply_offset():
        nonlocal rectify_H
        if base_H is None:
            return
        H = offset_matrix(*offset) @ base_H
        if rectify:
            rectify_H = scale @ H
            projector.set_warp_matrix(None)
        else:
            projector.set_warp_matrix(H)

    apply_offset()
    if base_H is not None:
        print(f"[projector] offset_x={offset[0]:.0f} offset_y={offset[1]:.0f}  (nudge: j/l/i/k)")

    show_window = bool(output_cfg.get("show_window", True))
    prev_time = time.time()
    smoothed_fps = None
    balls = None
    spawner = None

    print("Detection pipeline:", detection_pipeline.enabled_names)
    print("Object rules:", config.get("object_rules", {}))
    print("Quit: click the camera window and press q/Esc, or Ctrl+C here.")

    try:
        while True:
            ret, frame = source.read()
            if not ret:
                break

            if projector and not calib_size_checked and projector.calib_meta:
                expected = projector.calib_meta.get("camera_frame_size")
                actual = [frame.shape[1], frame.shape[0]]
                if expected and list(expected) != actual:
                    print(f"[projector] WARNING: calibrated for camera frame {expected}, got {actual}. Re-run calibrate_projector.py.")
                calib_size_checked = True

            if rectify_H is not None:
                frame = cv2.warpPerspective(frame, rectify_H, rectify_size)

            h, w = frame.shape[:2]
            if balls is None or spawner is None:
                balls = create_balls_from_config(config, frame_shape=frame.shape)
                spawner = BallSpawner(config["physics"])

            if spawner is not None:
                spawner.update_counter()
                max_balls = int(config["physics"].get("max_balls", 250))
                if spawner.should_spawn() and len(balls) < max_balls:
                    balls.append(spawner.spawn(w))

            obstacles, filter_detections, _filter_mask, occlusion_mask = detection_pipeline.detect(frame)
            if config.get("output", {}).get("show_person_mask", False):
                cv2.imshow("Person mask / occlusion mask", occlusion_mask)

            physics_cfg = config.get("physics", {})
            obstacle_iterations = int(physics_cfg.get("obstacle_collision_iterations", 1))

            alive_balls = []

            for ball in balls:
                ball.update(w, h)

                for _ in range(max(1, obstacle_iterations)):
                    for obstacle in obstacles:
                        resolve_ball_mask_collision(ball, obstacle)

                if not ball.dead:
                    alive_balls.append(ball)

            balls = alive_balls

            if physics_cfg.get("ball_collision_enabled", False):
                resolve_ball_ball_collisions(
                    balls,
                    iterations=int(physics_cfg.get("ball_collision_iterations", 2)),
                    restitution=float(physics_cfg.get("ball_restitution", 0.15)),
                    friction=float(physics_cfg.get("ball_friction", 0.92)),
                )

            now = time.time()
            instant_fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            smoothed_fps = instant_fps if smoothed_fps is None else 0.9 * smoothed_fps + 0.1 * instant_fps

            output = renderer.render(
                frame,
                obstacles,
                balls,
                fps=smoothed_fps,
                filter_detections=filter_detections,
                occlusion_mask=occlusion_mask,
            )

            if projector:
                if projector.content == "debug":
                    projector.display(output)
                else:
                    projector.display(
                        renderer.render_projection(frame.shape, balls, occlusion_mask=occlusion_mask)
                    )

            if show_window:
                cv2.imshow("Block Computing CV Mockup", output)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc (a program window must have focus)
                break
            nudges = {ord("j"): (-1, 0), ord("l"): (1, 0), ord("i"): (0, -1), ord("k"): (0, 1)}
            if key in nudges and base_H is not None:
                # j/l/i/k shift the camera->wall mapping left/right/up/down
                ddx, ddy = nudges[key]
                offset[0] += ddx * nudge_step
                offset[1] += ddy * nudge_step
                apply_offset()
                print(f"[projector] offset_x: {offset[0]:.0f}   offset_y: {offset[1]:.0f}   <- put these in config.yaml")
    except KeyboardInterrupt:
        print("Stopped with Ctrl+C.")
    finally:
        source.release()
        renderer.release()
        if projector:
            projector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()