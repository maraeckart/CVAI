import cv2
import numpy as np


COLORS = {
    "black_tape": (40, 40, 40),
    "yellow_tape": (0, 255, 255),
    "pink_postit": (255, 0, 255),
    "person_motion": (0, 255, 0),
    "person_yolo": (0, 255, 0),
}


class OverlayRenderer:
    def __init__(self, save_path: str | None = None, fps: float = 30.0, config: dict | None = None):
        self.save_path = save_path
        self.fps = fps
        self.writer = None
        output_cfg = (config or {}).get("output", {})
        self.draw_detections_enabled = bool(output_cfg.get("draw_detections", True))
        self.draw_person_filter = bool(output_cfg.get("draw_person_filter", False))
        self.hide_balls_behind_person = bool(output_cfg.get("hide_balls_behind_person", True))
        self.draw_person_mask_overlay = bool(output_cfg.get("draw_person_mask_overlay", False))

    def _ensure_writer(self, frame):
        if not self.save_path or self.writer is not None:
            return
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.save_path, fourcc, self.fps, (w, h))

    def draw_mask_overlay(self, frame, mask, color=(0, 255, 0), alpha=0.35):
        if mask is None:
            return frame

        overlay = frame.copy()
        overlay[mask > 0] = color

        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def draw_detections(self, frame, detections):
        if not self.draw_detections_enabled:
            return
        for det in detections:
            if det.is_filter_only and not self.draw_person_filter:
                continue
            color = COLORS.get(det.type, (0, 255, 0))
            if det.contour is not None:
                cv2.drawContours(frame, [det.contour], -1, color, 2)
                x, y, w, h = det.bbox
            else:
                x, y, w, h = det.bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                det.type,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    def draw_ball(self, frame, ball):
        center = tuple(ball.position.astype(int))
        radius = int(ball.radius)
        if ball.outline_color is not None and radius >= 3:
            cv2.circle(frame, center, radius + 1, ball.outline_color, -1)
        cv2.circle(frame, center, radius, ball.color, -1)

    def draw_balls(self, frame, balls, occlusion_mask=None):

        ball_layer = np.zeros_like(frame)
        ball_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for ball in balls:
            cx, cy = int(ball.position[0]), int(ball.position[1])
            r = int(ball.radius)

            # skip balls fully outside frame
            h, w = frame.shape[:2]
            if cx + r < 0 or cx - r >= w or cy + r < 0 or cy - r >= h:
                continue

            # draw outline if enabled
            if ball.outline_color is not None:
                cv2.circle(ball_layer, (cx, cy), r + 1, ball.outline_color, -1)
                cv2.circle(ball_mask, (cx, cy), r + 1, 255, -1)

            # draw fill
            cv2.circle(ball_layer, (cx, cy), r, ball.color, -1)
            cv2.circle(ball_mask, (cx, cy), r, 255, -1)

        # hide balls wherever the person/occlusion mask exists
        if occlusion_mask is not None:
            if occlusion_mask.shape[:2] != ball_mask.shape[:2]:
                occlusion_mask = cv2.resize(
                    occlusion_mask,
                    (ball_mask.shape[1], ball_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            ball_mask[occlusion_mask > 0] = 0

        frame[ball_mask > 0] = ball_layer[ball_mask > 0]

    def draw_hud(self, frame, detections, balls, fps=None):
        text = f"obstacles: {len(detections)} | balls: {len(balls)}"
        if fps is not None:
            text += f" | fps: {fps:.1f}"
        cv2.putText(
            frame,
            text,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def render(self, frame, detections, balls, fps=None, filter_detections=None, occlusion_mask=None):
        output = frame.copy()

        # Optional debug overlay first
        if self.draw_person_mask_overlay and occlusion_mask is not None:
            output = self.draw_mask_overlay(output, occlusion_mask)

        all_draw_detections = list(detections)

        if filter_detections:
            all_draw_detections.extend(filter_detections)

        self.draw_detections(output, all_draw_detections)

        # Balls must receive occlusion_mask here
        self.draw_balls(output, balls, occlusion_mask=occlusion_mask)

        self.draw_hud(output, detections, balls, fps=fps)

        self._ensure_writer(output)

        if self.writer is not None:
            self.writer.write(output)

        return output

    def release(self):
        if self.writer is not None:
            self.writer.release()
