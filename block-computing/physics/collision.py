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


def resolve_ball_mask_collision(ball: Ball, detection: Detection):
    """
    collision: circle against the actual obstacle mask/contour.
    Falls back to rectangle collision when no mask exists.
    """
    if detection.mask is None:
        return resolve_ball_rect_collision(ball, detection)

    bx, by = int(ball.position[0]), int(ball.position[1])
    r = int(ball.radius)
    h, w = detection.mask.shape[:2]

    if bx + r < 0 or bx - r >= w or by + r < 0 or by - r >= h:
        return False

    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (bx, by), r, 255, -1)
    overlap = cv2.bitwise_and(circle_mask, detection.mask)

    if cv2.countNonZero(overlap) == 0:
        return False

    # Approximate collision normal from the nearest obstacle-mask pixel to ball center.
    ys, xs = np.where(detection.mask > 0)
    if len(xs) == 0:
        return resolve_ball_rect_collision(ball, detection)

    dx = ball.position[0] - xs.astype(float)
    dy = ball.position[1] - ys.astype(float)
    nearest_idx = int(np.argmin(dx * dx + dy * dy))
    nearest = np.array([xs[nearest_idx], ys[nearest_idx]], dtype=float)
    normal = ball.position - nearest

    norm = float(np.linalg.norm(normal))
    if norm < 1e-6:
        normal = -ball.velocity.copy()
        norm = float(np.linalg.norm(normal))
    if norm < 1e-6:
        normal = np.array([0.0, -1.0])
    else:
        normal = normal / norm
    ball.position += normal * 2.0

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
