import random
import numpy as np
import cv2

# OpenCV uses BGR format
COLOR_MAP = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "orange": (0, 165, 255),
    "pink": (255, 0, 255),
    "turquoise": (255, 255, 0),
}


def _parse_color(value):
    if value is None:
        return None

    if isinstance(value, str):
        return COLOR_MAP.get(value.lower(), (255, 255, 255))

    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(v) for v in value)

    return (255, 255, 255)


class Ball:
    def __init__(
        self,
        position,
        velocity,
        radius=10,
        color=(255, 255, 255),
        outline_color=(0, 0, 0),
        gravity=0.0,
        damping=0.98,
        floor_friction=0.90,
        max_speed=15.0,
        lower_boundary_mode="bounce",
        obstacle_pushout=5.0,
        render_3d=False,  
    ):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.radius = int(radius)

        self.color = color
        self.outline_color = outline_color
        self.render_3d = bool(render_3d)  

        self.gravity = float(gravity)
        self.damping = float(damping)
        self.floor_friction = float(floor_friction)
        self.max_speed = float(max_speed)
        self.lower_boundary_mode = lower_boundary_mode

        self.dead = False
        self.resting = False
        self.obstacle_pushout = float(obstacle_pushout)

    @classmethod
    def from_config(cls, config):
        physics = config.get("physics", {})

        return cls(
            position=np.array(physics.get("initial_position", [120, 80]), dtype=float),
            velocity=np.array(physics.get("initial_velocity", [5.0, 3.5]), dtype=float),
            radius=int(physics.get("ball_radius", 8)),
            gravity=float(physics.get("gravity", 0.0)),
            damping=float(physics.get("bounce_damping", 0.98)),
            floor_friction=float(physics.get("floor_friction", 0.90)),
            max_speed=float(physics.get("max_speed", 15.0)),
            lower_boundary_mode=physics.get("lower_boundary_mode", "bounce"),
            color=_parse_color(physics.get("ball_color", "white")),
            outline_color=_parse_color(physics.get("ball_outline", "black")),
            render_3d=physics.get("render_3d", False),  # Parsed from config
        )

    def draw(self, frame):
        """
        Rates and renders the ball on the frame. Swaps styles based on config.
        """
        if self.dead:
            return

        cx, cy = int(self.position[0]), int(self.position[1])
        r = self.radius

        # outline
        if self.outline_color is not None and self.outline_color != "none":
            cv2.circle(frame, (cx, cy), r, self.outline_color, -1)

        # optional 3d rendering (has not worked yet)
        if self.render_3d:
            steps = max(5, int(r * 0.7))
            for i in range(steps):
                t = i / float(steps - 1)
            
                curr_b = int(self.color[0] + (255 - self.color[0]) * t)
                curr_g = int(self.color[1] + (255 - self.color[1]) * t)
                curr_r = int(self.color[2] + (255 - self.color[2]) * t)
                curr_color = (curr_b, curr_g, curr_r)

                curr_radius = int(r * (1.0 - t * 0.9))
                offset_x = int(cx - (r * 0.25) * t)
                offset_y = int(cy - (r * 0.25) * t)

                cv2.circle(frame, (offset_x, offset_y), curr_radius, curr_color, -1)
        
        # 2d mode
        else:
            cv2.circle(frame, (cx, cy), r, self.color, -1)

    def update(self, frame_width, frame_height):
        if self.dead:
            return

        self.velocity[1] += self.gravity

        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed

        self.position += self.velocity

        if self.position[0] <= self.radius:
            self.position[0] = self.radius
            self.velocity[0] = abs(self.velocity[0]) * self.damping

        elif self.position[0] >= frame_width - self.radius:
            self.position[0] = frame_width - self.radius
            self.velocity[0] = -abs(self.velocity[0]) * self.damping

        if self.position[1] <= self.radius:
            self.position[1] = self.radius
            self.velocity[1] = abs(self.velocity[1]) * self.damping

        bottom = frame_height - self.radius

        if self.lower_boundary_mode == "fall_through":
            if self.position[1] > frame_height + self.radius:
                self.dead = True
            return

        if self.lower_boundary_mode == "collect":
            if self.position[1] >= bottom:
                self.position[1] = bottom

                if abs(self.velocity[1]) > 0.8:
                    self.velocity[1] = -abs(self.velocity[1]) * self.damping
                else:
                    self.velocity[1] = 0.0
                    self.resting = True

                self.velocity[0] *= self.floor_friction

                if abs(self.velocity[0]) < 0.05:
                    self.velocity[0] = 0.0

            return

        if self.position[1] >= bottom:
            self.position[1] = bottom
            self.velocity[1] = -abs(self.velocity[1]) * self.damping


def create_balls_from_config(config, frame_shape=None):
    physics = config.get("physics", {})
    global_render_3d = physics.get("render_3d", False)

    explicit_balls = physics.get("balls")
    if explicit_balls:
        balls = []
        for item in explicit_balls:
            balls.append(
                Ball(
                    position=np.array(item.get("position", [120, 80]), dtype=float),
                    velocity=np.array(item.get("velocity", [5.0, 3.5]), dtype=float),
                    radius=int(item.get("radius", physics.get("ball_radius", 8))),
                    gravity=float(item.get("gravity", physics.get("gravity", 0.0))),
                    damping=float(item.get("bounce_damping", physics.get("bounce_damping", 0.98))),
                    floor_friction=float(item.get("floor_friction", physics.get("floor_friction", 0.90))),
                    max_speed=float(item.get("max_speed", physics.get("max_speed", 15.0))),
                    lower_boundary_mode=item.get(
                        "lower_boundary_mode",
                        physics.get("lower_boundary_mode", "bounce"),
                    ),
                    color=_parse_color(item.get("color", physics.get("ball_color", "white"))),
                    outline_color=_parse_color(item.get("outline", physics.get("ball_outline", "black"))),
                    render_3d=item.get("render_3d", global_render_3d), # Checked per individual item or falls back
                )
            )
        return balls

    count = int(physics.get("ball_count", 1))

    if count <= 1:
        return [Ball.from_config(config)]

    if frame_shape is None:
        frame_h, frame_w = 720, 1280
    else:
        frame_h, frame_w = frame_shape[:2]

    rng = random.Random(physics.get("random_seed", 7))

    radius = int(physics.get("ball_radius", 8))
    color = _parse_color(physics.get("ball_color", "white"))
    outline = _parse_color(physics.get("ball_outline", "black"))

    random_speed_min = float(physics.get("random_speed_min", 2.0))
    random_speed_max = float(physics.get("random_speed_max", 6.5))

    gravity = float(physics.get("gravity", 0.0))
    damping = float(physics.get("bounce_damping", 0.98))
    floor_friction = float(physics.get("floor_friction", 0.90))
    max_speed = float(physics.get("max_speed", 15.0))
    lower_boundary_mode = physics.get("lower_boundary_mode", "bounce")

    balls = []

    for _ in range(count):
        x = rng.uniform(radius, max(radius + 1, frame_w - radius))
        y = rng.uniform(radius, max(radius + 1, frame_h * 0.5))

        speed = rng.uniform(random_speed_min, random_speed_max)
        angle = rng.uniform(0, 2 * np.pi)

        vx = np.cos(angle) * speed
        vy = np.sin(angle) * speed

        balls.append(
            Ball(
                position=np.array([x, y], dtype=float),
                velocity=np.array([vx, vy], dtype=float),
                radius=radius,
                gravity=gravity,
                damping=damping,
                floor_friction=floor_friction,
                max_speed=max_speed,
                lower_boundary_mode=lower_boundary_mode,
                color=color,
                outline_color=outline,
                render_3d=global_render_3d, 
            )
        )

    return balls