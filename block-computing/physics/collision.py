import cv2
import numpy as np

from detection.base import Detection
from physics.ball import Ball


def clamp(value, low, high):
    return max(low, min(value, high))


def reflect_velocity(velocity: np.ndarray, normal: np.ndarray, damping: float = 0.98):
    normal = normal.astype(float)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return -velocity * damping
    normal = normal / norm
    return (velocity - 2 * np.dot(velocity, normal) * normal) * damping


def obstacle_field(detection: Detection):
    """Signed distance field of an obstacle mask, computed once per detection.

    sd > 0 outside (distance to the obstacle), sd < 0 inside (depth below its edge).
    The gradient of sd points away from the obstacle, so it is the collision normal
    everywhere, including deep inside the mask.
    """
    field = getattr(detection, "_field", None)
    if field is not None:
        return field

    inside = (detection.mask > 0).astype(np.uint8)
    dist_out = cv2.distanceTransform((1 - inside) * 255, cv2.DIST_L2, 3)
    dist_in = cv2.distanceTransform(inside * 255, cv2.DIST_L2, 3)
    sd = (dist_out - dist_in).astype(np.float32)
    sd = cv2.GaussianBlur(sd, (5, 5), 0)  # smoother normals on jagged YOLO edges
    gx = cv2.Sobel(sd, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sd, cv2.CV_32F, 0, 1, ksize=3)

    field = (sd, gx, gy)
    detection._field = field
    return field


def resolve_ball_mask_collision(ball: Ball, detection: Detection):
    """
    Circle against the obstacle mask, using a signed distance field.
    Falls back to rectangle collision when no mask exists.
    """
    if detection.mask is None:
        return resolve_ball_rect_collision(ball, detection)

    h, w = detection.mask.shape[:2]
    r = float(ball.radius)
    bx, by = float(ball.position[0]), float(ball.position[1])

    if bx + r < 0 or bx - r >= w or by + r < 0 or by - r >= h:
        return False

    sd, gx, gy = obstacle_field(detection)
    ix = int(clamp(round(bx), 0, w - 1))
    iy = int(clamp(round(by), 0, h - 1))

    dist = float(sd[iy, ix])
    if dist >= r:
        return False

    normal = np.array([gx[iy, ix], gy[iy, ix]], dtype=float)
    norm = float(np.linalg.norm(normal))
    normal = normal / norm if norm > 1e-6 else np.array([0.0, -1.0])

    # Push out by the real overlap (+ small margin). Capped per call so a ball
    # swept deep into the body leaves it over a few iterations instead of teleporting.
    penetration = r - dist
    pushout = float(getattr(ball, "obstacle_pushout", 2.0))
    ball.position += normal * min(penetration + pushout, r * 1.5)

    if np.dot(ball.velocity, normal) < 0:
        ball.velocity = reflect_velocity(ball.velocity, normal, ball.damping)

    return True


def resolve_ball_rect_collision(ball: Ball, detection: Detection):
    """Circle-vs-axis-aligned-rectangle collision fallback."""
    x1, y1, x2, y2 = detection.rect_xyxy
    bx, by = ball.position
    r = ball.radius

    closest_x = clamp(bx, x1, x2)
    closest_y = clamp(by, y1, y2)

    delta = np.array([bx - closest_x, by - closest_y], dtype=float)
    dist_sq = float(np.dot(delta, delta))

    if dist_sq > r * r:
        return False

    pushout = float(getattr(ball, "obstacle_pushout", 2.0))

    if dist_sq == 0:
        distances = {
            "left": abs(bx - x1),
            "right": abs(x2 - bx),
            "top": abs(by - y1),
            "bottom": abs(y2 - by),
        }

        side = min(distances, key=distances.get)

        if side == "left":
            normal = np.array([-1.0, 0.0])
            ball.position[0] = x1 - r - pushout
        elif side == "right":
            normal = np.array([1.0, 0.0])
            ball.position[0] = x2 + r + pushout
        elif side == "top":
            normal = np.array([0.0, -1.0])
            ball.position[1] = y1 - r - pushout
        else:
            normal = np.array([0.0, 1.0])
            ball.position[1] = y2 + r + pushout

    else:
        distance = np.sqrt(dist_sq)
        normal = delta / distance
        penetration = r - distance

        # Important: push out by actual overlap + small safety margin.
        ball.position += normal * (penetration + pushout)

    if np.dot(ball.velocity, normal) < 0:
        ball.velocity = reflect_velocity(ball.velocity, normal, ball.damping)

    return True
