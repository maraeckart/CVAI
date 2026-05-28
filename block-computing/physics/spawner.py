import random

from physics.ball import Ball, _parse_color


class BallSpawner:
    def __init__(self, physics_cfg):
        self.cfg = physics_cfg
        self.frame_counter = 0

        seed = self.cfg.get("random_seed", None)
        self.rng = random.Random(seed)

    def update_counter(self):
        self.frame_counter += 1

    def should_spawn(self):
        gravity = float(self.cfg.get("gravity", 0.0))

        if gravity == 0.0:
            return False

        if not self.cfg.get("spawn_enabled", False):
            return False

        interval = int(self.cfg.get("spawn_interval_frames", 10))

        return self.frame_counter % max(1, interval) == 0

    def spawn(self, frame_width):
        radius = int(self.cfg.get("ball_radius", 7))

        x_min = self.cfg.get("spawn_x_min", radius)
        x_max = self.cfg.get("spawn_x_max", None)

        if x_max is None:
            x_max = frame_width - radius

        x = self.rng.uniform(float(x_min), float(x_max))
        y = float(self.cfg.get("spawn_y", -radius * 2))

        vx = self.rng.uniform(
            float(self.cfg.get("spawn_vx_min", -1.0)),
            float(self.cfg.get("spawn_vx_max", 1.0)),
        )

        vy = self.rng.uniform(
            float(self.cfg.get("spawn_vy_min", 0.0)),
            float(self.cfg.get("spawn_vy_max", 1.5)),
        )

        return Ball(
            position=[x, y],
            velocity=[vx, vy],
            radius=radius,
            color=_parse_color(self.cfg.get("ball_color", "white")),
            outline_color=_parse_color(self.cfg.get("ball_outline", "black")),
            gravity=float(self.cfg.get("gravity", 0.0)),
            damping=float(self.cfg.get("bounce_damping", 0.72)),
            floor_friction=float(self.cfg.get("floor_friction", 0.90)),
            max_speed=float(self.cfg.get("max_speed", 15.0)),
            lower_boundary_mode=self.cfg.get("lower_boundary_mode", "collect"),
        )