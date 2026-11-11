import numpy as np


def resolve_ball_ball_collisions(
    balls,
    iterations=2,
    restitution=0.15,
    friction=0.92,
):
    """
    Simple equal-mass circle-vs-circle collision solver.

    - Separates overlapping balls.
    - Adjusts velocities so they do not pass through each other.
    - Low restitution makes piles/heaps instead of bouncy gas.
    """

    if len(balls) < 2:
        return

    for _ in range(max(1, int(iterations))):
        for i in range(len(balls)):
            a = balls[i]

            if getattr(a, "dead", False):
                continue

            for j in range(i + 1, len(balls)):
                b = balls[j]

                if getattr(b, "dead", False):
                    continue

                delta = b.position - a.position
                dist_sq = float(np.dot(delta, delta))

                min_dist = float(a.radius + b.radius)

                if dist_sq >= min_dist * min_dist:
                    continue

                if dist_sq < 1e-8:
                    # Avoid divide-by-zero when two balls are exactly on top of each other.
                    normal = np.array([1.0, 0.0], dtype=float)
                    dist = 0.0
                else:
                    dist = float(np.sqrt(dist_sq))
                    normal = delta / dist

                overlap = min_dist - dist

                # Positional correction: split the overlap equally.
                correction = normal * (overlap * 0.5)

                a.position -= correction
                b.position += correction

                # Velocity correction.
                relative_velocity = b.velocity - a.velocity
                velocity_along_normal = float(np.dot(relative_velocity, normal))

                # If they are already separating, only positional correction was needed.
                if velocity_along_normal > 0:
                    continue

                # Equal mass impulse.
                impulse_scalar = -(1.0 + restitution) * velocity_along_normal
                impulse_scalar /= 2.0

                impulse = impulse_scalar * normal

                a.velocity -= impulse
                b.velocity += impulse

                # Tangential friction for heap-like behavior. It must act on the
                # RELATIVE sliding velocity only: damping each ball's absolute
                # velocity made a whole clump brake itself while falling, so it
                # sank slowly instead of falling ("floating balls").
                tangent = np.array([-normal[1], normal[0]], dtype=float)
                rel_t = float(np.dot(b.velocity - a.velocity, tangent))
                friction_impulse = tangent * rel_t * (1.0 - friction) * 0.5

                a.velocity += friction_impulse
                b.velocity -= friction_impulse