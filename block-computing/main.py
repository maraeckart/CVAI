import time
from pathlib import Path

import cv2
import yaml

from detection.pipeline import DetectionPipeline
from input.camera_source import CameraSource
from input.video_source import VideoSource
from physics.ball import create_balls_from_config
from physics.collision import resolve_ball_mask_collision
from rendering.overlay import OverlayRenderer
from physics.spawner import BallSpawner
from physics.particle_collusion import resolve_ball_ball_collisions


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
        return CameraSource(input_cfg.get("camera_index", 0), resize_width=resize_width)
    raise ValueError(f"Unknown input mode: {mode}")


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

    show_window = bool(output_cfg.get("show_window", True))
    prev_time = time.time()
    smoothed_fps = None
    balls = None
    spawner = None

    print("Detection pipeline:", detection_pipeline.enabled_names)
    print("Object rules:", config.get("object_rules", {}))
    print("Press q to quit.")

    try:
        while True:
            ret, frame = source.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if balls is None or spawner is None:
                balls = create_balls_from_config(config, frame_shape=frame.shape)
                spawner = BallSpawner(config["physics"])
            
            if spawner is not None:
                spawner.update_counter()

                max_balls = int(config["physics"].get("max_balls", 250))

                if spawner.should_spawn() and len(balls) < max_balls:
                    balls.append(spawner.spawn(w))

            # Unified pass:
            # - YOLO person segmentation creates filter/occlusion masks.
            # - HSV detectors detect colored physical blocks outside the person mask.
            # - object_rules decide what becomes a physics obstacle.
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
            physics_cfg = config.get("physics", {})

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

            if show_window:
                cv2.imshow("Block Computing CV Mockup", output)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    finally:
        source.release()
        renderer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
