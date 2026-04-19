"""
core/side_by_side_renderer.py — Professional comparison video renderer
Design: publication-quality, dark theme, NeurIPS/AAAI paper style
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from models import HumanPose, Keypoint3D, RetargetedPose
from config import config

# ── Design constants ──────────────────────────────────────────────────────────
W = 1280
H = 720
PANEL_W = 620
PANEL_H = 580
PANEL_Y = 80          # top offset for header
INFO_H = 60           # bottom info bar height

# Colors — dark professional theme
C_BG         = (10,  10,  18)    # near-black background
C_HEADER_BG  = (15,  15,  25)    # header
C_PANEL_BG   = (18,  18,  30)    # panel background
C_DIVIDER    = (40,  40,  60)    # subtle divider
C_ACCENT_L   = (64,  186, 255)   # left accent — cyan (human)
C_ACCENT_R   = (100, 220, 130)   # right accent — green (robot)
C_ACCENT_W   = (255, 180,  60)   # warning/highlight — amber
C_TEXT_PRI   = (230, 230, 240)   # primary text
C_TEXT_SEC   = (130, 130, 155)   # secondary text
C_TEXT_DIM   = (70,  70,  90)    # dim text
C_SKELETON   = (64,  186, 255)   # skeleton cyan
C_JOINT      = (255, 255, 255)   # joint dots
C_HAND       = (255, 200,  80)   # hand landmarks amber

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO  = cv2.FONT_HERSHEY_PLAIN

# Skeleton connections
SKELETON_PAIRS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("nose",           "left_shoulder"),
    ("nose",           "right_shoulder"),
]


class SideBySideRenderer:
    """
    Professional comparison video renderer.
    Layout:
      ┌────────────────────────────────────────────┐
      │  [Header bar — title + model info]          │ 80px
      ├──────────────────┬─────┬───────────────────┤
      │  Human Demo      │  │  │  Unitree H1       │ 580px
      │  (video+skeleton)│  │  │  (robot HUD)      │
      ├──────────────────┴─────┴───────────────────┤
      │  [Metrics bar — live scores + timeline]     │ 60px
      └────────────────────────────────────────────┘
    """

    def __init__(self, output_path: str, task: str = "", fps: float = 15.0):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.fps = fps
        self.frame_count = 0
        self.total_frames = 0   # set later for progress bar
        self._metric_history: list[dict] = []

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (W, H)
        )
        print(f"[Renderer] Output: {self.output_path} ({W}x{H} @ {fps}fps)")

    def set_total_frames(self, n: int):
        self.total_frames = n

    def write_frame(
        self,
        human_frame: Optional[np.ndarray],
        robot_frame: Optional[np.ndarray],
        human_pose: Optional[HumanPose] = None,
        retargeted_pose: Optional[RetargetedPose] = None,
        extra_metrics: dict = None,
    ):
        canvas = self._make_canvas()
        self._draw_header(canvas)
        self._draw_human_panel(canvas, human_frame, human_pose)
        self._draw_robot_panel(canvas, robot_frame, retargeted_pose)
        self._draw_divider(canvas)
        self._draw_metrics_bar(canvas, retargeted_pose, extra_metrics)
        self._draw_progress(canvas)

        self.writer.write(canvas)
        self.frame_count += 1

        # Track metric history for trend lines
        if retargeted_pose:
            self._metric_history.append({
                "balance": retargeted_pose.balance_score,
                "error": retargeted_pose.retargeting_error,
            })

    def write_batch(
        self,
        human_frames: list,
        robot_frames: list,
        poses: list = None,
        retargeted: list = None,
        metrics_list: list = None,
    ):
        self.set_total_frames(max(len(human_frames), len(robot_frames)))
        n = min(len(human_frames), len(robot_frames))
        for i in range(n):
            self.write_frame(
                human_frame=human_frames[i],
                robot_frame=robot_frames[i],
                human_pose=poses[i] if poses and i < len(poses) else None,
                retargeted_pose=retargeted[i] if retargeted and i < len(retargeted) else None,
                extra_metrics=metrics_list[i] if metrics_list and i < len(metrics_list) else None,
            )
        print(f"[Renderer] Wrote {n} frames")

    def add_title_card(
        self,
        title: str,
        subtitle: str = "",
        duration_seconds: float = 2.0,
    ):
        """Professional title card with gradient-like effect."""
        n = int(duration_seconds * self.fps)
        for fi in range(n):
            alpha = min(1.0, fi / max(int(self.fps * 0.3), 1))  # fade in
            canvas = self._make_canvas()

            # Center content
            self._draw_title_text(canvas, title, subtitle, alpha)
            self._draw_corner_accents(canvas, alpha)

            self.writer.write(canvas)

    def finalize(self) -> str:
        self.writer.release()
        size_mb = self.output_path.stat().st_size / 1e6
        print(f"[Renderer] Finalized: {self.frame_count} frames → "
              f"{self.output_path} ({size_mb:.1f} MB)")
        return str(self.output_path)

    # ── Canvas construction ────────────────────────────────────────────────────

    def _make_canvas(self) -> np.ndarray:
        canvas = np.full((H, W, 3), C_BG, dtype=np.uint8)
        return canvas

    def _draw_header(self, canvas: np.ndarray):
        """Top header bar with title and model info."""
        cv2.rectangle(canvas, (0, 0), (W, PANEL_Y - 2), C_HEADER_BG, -1)
        cv2.line(canvas, (0, PANEL_Y - 2), (W, PANEL_Y - 2), C_DIVIDER, 1)

        # Paper title (left)
        title = "LLM-Guided Kinematic Retargeting"
        cv2.putText(canvas, title, (20, 30), FONT_BOLD, 0.65, C_TEXT_PRI, 1)

        # Task (center)
        task_short = self.task[:55] + "..." if len(self.task) > 55 else self.task
        task_display = f"Task: {task_short}"
        tw = cv2.getTextSize(task_display, FONT, 0.45, 1)[0][0]
        cv2.putText(canvas, task_display, (W//2 - tw//2, 55),
                    FONT, 0.45, C_TEXT_SEC, 1)

        # Model badge (right)
        model = config.openrouter.reasoning_model.split("/")[-1]
        badge_text = f"  {model}  "
        bw = cv2.getTextSize(badge_text, FONT, 0.40, 1)[0][0]
        bx = W - bw - 20
        cv2.rectangle(canvas, (bx - 4, 14), (bx + bw + 4, 36),
                      (30, 50, 80), -1)
        cv2.rectangle(canvas, (bx - 4, 14), (bx + bw + 4, 36),
                      C_ACCENT_L, 1)
        cv2.putText(canvas, badge_text, (bx, 30),
                    FONT, 0.40, C_ACCENT_L, 1)

        # Frame counter (right, below badge)
        fc_text = f"Frame {self.frame_count:04d}"
        cv2.putText(canvas, fc_text, (W - 110, 56),
                    FONT_MONO, 0.9, C_TEXT_DIM, 1)

    def _draw_human_panel(
        self,
        canvas: np.ndarray,
        frame: Optional[np.ndarray],
        pose: Optional[HumanPose],
    ):
        """Left panel: human video with clean skeleton overlay."""
        px, py = 10, PANEL_Y
        pw, ph = PANEL_W, PANEL_H

        # Panel background
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), C_PANEL_BG, -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), C_DIVIDER, 1)

        # Video frame
        if frame is not None and frame.size > 0:
            fh_orig, fw_orig = frame.shape[:2]
            inner_w, inner_h = pw - 2, ph - 40
            img = self._fit_frame(frame, inner_w, inner_h)
            ih, iw = img.shape[:2]
            ix = px + 1 + (inner_w - iw) // 2
            iy = py + 1

            # Skeleton: MediaPipe coords are normalized to the *source* frame, not the letterbox
            if pose:
                img = self._draw_skeleton_on(
                    img, pose, iw, ih, fw_orig, fh_orig
                )

            canvas[iy:iy + ih, ix:ix + iw] = img
        else:
            # No frame — show placeholder
            cx, cy = px + pw // 2, py + ph // 2 - 20
            cv2.putText(canvas, "No video frame", (cx - 70, cy),
                        FONT, 0.5, C_TEXT_DIM, 1)

        # Panel label bar (bottom of panel)
        label_y = py + ph - 38
        cv2.rectangle(canvas, (px, label_y), (px + pw, py + ph),
                      (12, 20, 35), -1)
        cv2.line(canvas, (px, label_y), (px + pw, label_y), C_ACCENT_L, 2)

        # Left accent stripe
        cv2.rectangle(canvas, (px, py), (px + 3, py + ph), C_ACCENT_L, -1)

        # Label
        cv2.putText(canvas, "HUMAN DEMONSTRATION",
                    (px + 14, py + ph - 18), FONT_BOLD, 0.50, C_ACCENT_L, 1)

        # Mode badge
        if pose:
            mode = pose.dominant_mode.value.upper()
            mode_colors = {
                "LOCOMOTION":   (255, 140,  50),
                "MANIPULATION": (100, 200, 255),
                "WHOLE_BODY":   (180, 100, 255),
            }
            mc = mode_colors.get(mode, C_TEXT_SEC)
            mw = cv2.getTextSize(mode, FONT, 0.38, 1)[0][0]
            mx = px + pw - mw - 18
            cv2.rectangle(canvas, (mx - 6, py + ph - 32),
                          (mx + mw + 6, py + ph - 12), (20, 20, 30), -1)
            cv2.rectangle(canvas, (mx - 6, py + ph - 32),
                          (mx + mw + 6, py + ph - 12), mc, 1)
            cv2.putText(canvas, mode, (mx, py + ph - 17),
                        FONT, 0.38, mc, 1)

    def _draw_robot_panel(
        self,
        canvas: np.ndarray,
        frame: Optional[np.ndarray],
        retargeted: Optional[RetargetedPose],
    ):
        """Right panel: robot HUD with joint visualization."""
        px = W - PANEL_W - 10
        py = PANEL_Y
        pw, ph = PANEL_W, PANEL_H

        # Panel background
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), C_PANEL_BG, -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), C_DIVIDER, 1)

        # Try real robot frame first
        has_real_frame = (
            frame is not None and frame.size > 0 and frame.mean() > 15
        )

        if has_real_frame:
            img = self._fit_frame(frame, pw - 2, ph - 40)
            ih, iw = img.shape[:2]
            ix = px + 1 + (pw - 2 - iw) // 2
            iy = py + 1
            canvas[iy:iy + ih, ix:ix + iw] = img
        else:
            # Professional robot HUD visualization
            self._draw_robot_hud(canvas, px, py, pw, ph, retargeted)

        # Panel label bar
        label_y = py + ph - 38
        cv2.rectangle(canvas, (px, label_y), (px + pw, py + ph),
                      (12, 30, 20), -1)
        cv2.line(canvas, (px, label_y), (px + pw, label_y), C_ACCENT_R, 2)

        # Right accent stripe
        cv2.rectangle(canvas, (px + pw - 3, py),
                      (px + pw, py + ph), C_ACCENT_R, -1)

        cv2.putText(canvas, "UNITREE H1  ·  GENESIS SIMULATION",
                    (px + 14, py + ph - 18), FONT_BOLD, 0.50, C_ACCENT_R, 1)

        # Balance score mini-bar
        if retargeted:
            b = retargeted.balance_score
            bar_x = px + pw - 120
            bar_y = py + ph - 30
            bar_w = 100
            cv2.rectangle(canvas, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + 10), (20, 35, 25), -1)
            fill_color = C_ACCENT_R if b > 0.6 else C_ACCENT_W
            cv2.rectangle(canvas, (bar_x, bar_y),
                          (bar_x + int(bar_w * b), bar_y + 10),
                          fill_color, -1)
            cv2.putText(canvas, f"BAL {b:.2f}",
                        (bar_x - 52, bar_y + 9), FONT, 0.38, C_TEXT_SEC, 1)

    def _draw_robot_hud(
        self,
        canvas: np.ndarray,
        px: int, py: int, pw: int, ph: int,
        retargeted: Optional[RetargetedPose],
    ):
        """
        Professional robot HUD when Genesis can't render.
        Shows stick figure + joint angle bars + status.
        """
        cx = px + pw // 2
        cy = py + ph // 2 - 30

        # ── Stick figure robot ─────────────────────────────────────────────
        angles = retargeted.joint_angles if retargeted else {}

        # Get joint angles to animate figure
        l_elbow = angles.get("left_elbow", 0.4)
        r_elbow = angles.get("right_elbow", 0.4)
        l_sp = angles.get("left_shoulder_pitch", 0.0)
        r_sp = angles.get("right_shoulder_pitch", 0.0)
        l_knee = angles.get("left_knee", 0.62)
        r_knee = angles.get("right_knee", 0.62)

        # Scale for figure
        scale = 80
        lw = 2   # line width

        # Head
        cv2.circle(canvas, (cx, cy - scale - 15), 18,
                   C_ACCENT_R, 2)
        # Visor line (robot face)
        cv2.line(canvas, (cx - 10, cy - scale - 12),
                 (cx + 10, cy - scale - 12), C_ACCENT_R, 2)

        # Torso
        torso_top = (cx, cy - scale + 5)
        torso_bot = (cx, cy)
        cv2.line(canvas, torso_top, torso_bot, C_ACCENT_R, lw + 1)

        # Shoulders
        l_shoulder = (cx - 35, cy - scale + 15)
        r_shoulder = (cx + 35, cy - scale + 15)
        cv2.line(canvas, l_shoulder, r_shoulder, C_ACCENT_R, lw)

        # Left arm (animated)
        import math
        la_len = 45
        la_angle = -math.pi/2 - l_sp - 0.3
        l_elbow_pos = (
            int(l_shoulder[0] + la_len * math.cos(la_angle)),
            int(l_shoulder[1] + la_len * math.sin(la_angle)),
        )
        cv2.line(canvas, l_shoulder, l_elbow_pos, C_ACCENT_R, lw)

        la2_angle = la_angle + l_elbow + 0.3
        l_wrist_pos = (
            int(l_elbow_pos[0] + la_len * 0.85 * math.cos(la2_angle)),
            int(l_elbow_pos[1] + la_len * 0.85 * math.sin(la2_angle)),
        )
        cv2.line(canvas, l_elbow_pos, l_wrist_pos, C_ACCENT_R, lw)

        # Right arm (animated, mirrored)
        ra_angle = -math.pi/2 + r_sp + 0.3
        r_elbow_pos = (
            int(r_shoulder[0] + la_len * math.cos(ra_angle)),
            int(r_shoulder[1] + la_len * math.sin(ra_angle)),
        )
        cv2.line(canvas, r_shoulder, r_elbow_pos, C_ACCENT_R, lw)

        ra2_angle = ra_angle - r_elbow - 0.3
        r_wrist_pos = (
            int(r_elbow_pos[0] + la_len * 0.85 * math.cos(ra2_angle)),
            int(r_elbow_pos[1] + la_len * 0.85 * math.sin(ra2_angle)),
        )
        cv2.line(canvas, r_elbow_pos, r_wrist_pos, C_ACCENT_R, lw)

        # Hips
        l_hip = (cx - 20, cy)
        r_hip = (cx + 20, cy)
        cv2.line(canvas, l_hip, r_hip, C_ACCENT_R, lw)

        # Legs (animated)
        ll_len = 55
        ll_angle = math.pi/2 + min(l_knee * 0.3, 0.4)
        l_knee_pos = (
            int(l_hip[0] - 8 + ll_len * math.cos(math.pi/2 + 0.1)),
            int(l_hip[1] + ll_len * math.sin(math.pi/2 + 0.1)),
        )
        cv2.line(canvas, l_hip, l_knee_pos, C_ACCENT_R, lw)
        l_ankle_pos = (
            int(l_knee_pos[0] + ll_len * math.cos(math.pi/2 - ll_angle * 0.2)),
            int(l_knee_pos[1] + ll_len * math.sin(math.pi/2 - ll_angle * 0.2)),
        )
        cv2.line(canvas, l_knee_pos, l_ankle_pos, C_ACCENT_R, lw)

        rr_angle = math.pi/2 + min(r_knee * 0.3, 0.4)
        r_knee_pos = (
            int(r_hip[0] + 8 + ll_len * math.cos(math.pi/2 - 0.1)),
            int(r_hip[1] + ll_len * math.sin(math.pi/2 - 0.1)),
        )
        cv2.line(canvas, r_hip, r_knee_pos, C_ACCENT_R, lw)
        r_ankle_pos = (
            int(r_knee_pos[0] + ll_len * math.cos(math.pi/2 + rr_angle * 0.2)),
            int(r_knee_pos[1] + ll_len * math.sin(math.pi/2 + rr_angle * 0.2)),
        )
        cv2.line(canvas, r_knee_pos, r_ankle_pos, C_ACCENT_R, lw)

        # Joint dots
        for pt in [l_shoulder, r_shoulder, l_elbow_pos, r_elbow_pos,
                   l_wrist_pos, r_wrist_pos, l_hip, r_hip,
                   l_knee_pos, r_knee_pos]:
            cv2.circle(canvas, pt, 4, (30, 30, 50), -1)
            cv2.circle(canvas, pt, 4, C_ACCENT_R, 1)

        # Ground line
        ground_y = py + ph - 60
        cv2.line(canvas, (px + 20, ground_y), (px + pw - 20, ground_y),
                 C_DIVIDER, 1)

        # ── Joint angle bars (right side) ──────────────────────────────────
        bar_x = px + pw - 200
        bar_y_start = py + 30
        bar_w = 150
        bar_h = 10

        key_joints = [
            ("L.Shoulder", "left_shoulder_pitch",  (-3.14, 3.14)),
            ("R.Shoulder", "right_shoulder_pitch", (-3.14, 3.14)),
            ("L.Elbow",    "left_elbow",            (-1.57, 1.57)),
            ("R.Elbow",    "right_elbow",            (-1.57, 1.57)),
            ("L.Knee",     "left_knee",              (-0.26, 2.05)),
            ("R.Knee",     "right_knee",             (-0.26, 2.05)),
            ("Torso",      "torso",                  (-2.35, 2.35)),
        ]

        cv2.putText(canvas, "JOINT ANGLES (rad)",
                    (bar_x, bar_y_start - 10), FONT, 0.36, C_TEXT_DIM, 1)

        for i, (label, joint, (lo, hi)) in enumerate(key_joints):
            val = angles.get(joint, 0.0)
            by = bar_y_start + i * 26

            # Label
            cv2.putText(canvas, label, (bar_x, by + 8),
                        FONT_MONO, 0.80, C_TEXT_SEC, 1)

            # Track
            tx = bar_x + 75
            cv2.rectangle(canvas, (tx, by), (tx + bar_w, by + bar_h),
                          (25, 25, 40), -1)

            # Zero marker
            zero_x = tx + int(bar_w * (-lo) / (hi - lo))
            cv2.line(canvas, (zero_x, by - 2), (zero_x, by + bar_h + 2),
                     C_TEXT_DIM, 1)

            # Value bar
            norm = (val - lo) / (hi - lo)
            norm = max(0.0, min(1.0, norm))
            fill_x = int(bar_w * norm)
            bar_color = C_ACCENT_R if abs(val) < (hi * 0.7) else C_ACCENT_W
            cv2.rectangle(canvas, (tx, by), (tx + fill_x, by + bar_h),
                          bar_color, -1)

            # Value text
            cv2.putText(canvas, f"{val:+.2f}",
                        (tx + bar_w + 6, by + 8),
                        FONT_MONO, 0.75, C_TEXT_SEC, 1)

        # ── Status badges ───────────────────────────────────────────────────
        status_y = py + ph - 110
        if retargeted:
            tags = [
                (f"BAL:{retargeted.balance_score:.2f}",
                 C_ACCENT_R if retargeted.balance_score > 0.6 else C_ACCENT_W),
                (f"REACH:{retargeted.reachability_score:.2f}", C_ACCENT_L),
                (f"ERR:{retargeted.retargeting_error:.3f}", C_TEXT_SEC),
            ]
            tx = px + 14
            for tag, color in tags:
                tw = cv2.getTextSize(tag, FONT, 0.38, 1)[0][0]
                cv2.rectangle(canvas, (tx - 4, status_y - 12),
                              (tx + tw + 4, status_y + 4),
                              (20, 20, 35), -1)
                cv2.rectangle(canvas, (tx - 4, status_y - 12),
                              (tx + tw + 4, status_y + 4), color, 1)
                cv2.putText(canvas, tag, (tx, status_y),
                            FONT, 0.38, color, 1)
                tx += tw + 18

        # Model watermark bottom-left of panel
        cv2.putText(canvas, "LLM-Retargeted  ·  No Training",
                    (px + 14, py + ph - 50), FONT, 0.35, C_TEXT_DIM, 1)

    def _draw_divider(self, canvas: np.ndarray):
        """Center vertical divider with VS label."""
        dx = W // 2
        cv2.line(canvas, (dx, PANEL_Y), (dx, PANEL_Y + PANEL_H),
                 C_DIVIDER, 1)
        # VS badge
        vs_y = PANEL_Y + PANEL_H // 2
        cv2.rectangle(canvas, (dx - 18, vs_y - 14),
                      (dx + 18, vs_y + 14), C_BG, -1)
        cv2.rectangle(canvas, (dx - 18, vs_y - 14),
                      (dx + 18, vs_y + 14), C_DIVIDER, 1)
        cv2.putText(canvas, "VS", (dx - 11, vs_y + 5),
                    FONT_BOLD, 0.50, C_TEXT_DIM, 1)

    def _draw_metrics_bar(
        self,
        canvas: np.ndarray,
        retargeted: Optional[RetargetedPose],
        extra: Optional[dict],
    ):
        """Bottom metrics bar with live scores."""
        by = PANEL_Y + PANEL_H + 2
        bh = H - by

        cv2.rectangle(canvas, (0, by), (W, H), C_HEADER_BG, -1)
        cv2.line(canvas, (0, by), (W, by), C_DIVIDER, 1)

        # Metric items
        metrics = []
        if retargeted:
            metrics += [
                ("Naturalness", extra.get("naturalness", 0.0) if extra else 0.0,
                 C_ACCENT_L),
                ("Balance", retargeted.balance_score, C_ACCENT_R),
                ("Ret.Error", 1.0 - retargeted.retargeting_error, C_ACCENT_W),
            ]
        if extra:
            if "stability" in extra:
                metrics.append(("Stability", extra["stability"], C_ACCENT_R))

        # Draw metric items evenly spaced
        item_w = W // max(len(metrics) + 2, 5)
        for i, (label, val, color) in enumerate(metrics):
            ix = 20 + i * (item_w + 10)
            iy = by + 8

            # Mini bar
            bar_w = 80
            cv2.rectangle(canvas, (ix, iy + 14),
                          (ix + bar_w, iy + 22), (25, 25, 40), -1)
            cv2.rectangle(canvas, (ix, iy + 14),
                          (ix + int(bar_w * max(0, min(1, val))), iy + 22),
                          color, -1)

            cv2.putText(canvas, label, (ix, iy + 10),
                        FONT, 0.38, C_TEXT_SEC, 1)
            cv2.putText(canvas, f"{val:.2f}", (ix + bar_w + 4, iy + 22),
                        FONT_MONO, 0.90, color, 1)

        # Frame counter (right)
        fc = f"Frame {self.frame_count:04d}"
        if self.total_frames > 0:
            fc += f" / {self.total_frames:04d}"
        cv2.putText(canvas, fc, (W - 180, by + 30),
                    FONT_MONO, 1.0, C_TEXT_DIM, 1)

    def _draw_progress(self, canvas: np.ndarray):
        """Thin progress timeline at very bottom."""
        if self.total_frames <= 0:
            return
        prog = self.frame_count / self.total_frames
        py = H - 4
        cv2.rectangle(canvas, (0, py), (W, H), C_DIVIDER, -1)
        cv2.rectangle(canvas, (0, py),
                      (int(W * prog), H), C_ACCENT_L, -1)

    def _draw_title_text(
        self,
        canvas: np.ndarray,
        title: str,
        subtitle: str,
        alpha: float,
    ):
        """Title card content."""
        # Subtle grid pattern
        for x in range(0, W, 40):
            cv2.line(canvas, (x, 0), (x, H), (15, 15, 25), 1)
        for y in range(0, H, 40):
            cv2.line(canvas, (0, y), (W, y), (15, 15, 25), 1)

        # Accent lines
        acc_y = H // 2 - 50
        cv2.line(canvas, (W//4, acc_y - 1), (3*W//4, acc_y - 1),
                 C_ACCENT_L, 1)
        cv2.line(canvas, (W//4, acc_y + 80), (3*W//4, acc_y + 80),
                 C_ACCENT_R, 1)

        # Main title
        tw = cv2.getTextSize(title, FONT_BOLD, 1.1, 2)[0][0]
        tx = (W - tw) // 2
        color_t = tuple(int(c * alpha) for c in C_TEXT_PRI)
        cv2.putText(canvas, title, (tx, acc_y + 40),
                    FONT_BOLD, 1.1, color_t, 2)

        # Subtitle
        if subtitle:
            sw = cv2.getTextSize(subtitle, FONT, 0.60, 1)[0][0]
            sx = (W - sw) // 2
            color_s = tuple(int(c * alpha) for c in C_TEXT_SEC)
            cv2.putText(canvas, subtitle, (sx, acc_y + 72),
                        FONT, 0.60, color_s, 1)

        # Corner labels
        color_c = tuple(int(c * alpha) for c in C_ACCENT_L)
        color_cr = tuple(int(c * alpha) for c in C_ACCENT_R)
        cv2.putText(canvas, "HUMAN DEMO", (60, H//2 + 140),
                    FONT, 0.50, color_c, 1)
        cv2.putText(canvas, "UNITREE H1", (W - 200, H//2 + 140),
                    FONT, 0.50, color_cr, 1)

    def _draw_corner_accents(self, canvas: np.ndarray, alpha: float):
        """Decorative corner brackets."""
        size = 30
        thick = 2
        color = tuple(int(c * alpha) for c in C_DIVIDER)
        corners = [(20, 20), (W - 20, 20), (20, H - 20), (W - 20, H - 20)]
        dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        for (cx, cy), (dx, dy) in zip(corners, dirs):
            cv2.line(canvas, (cx, cy), (cx + dx * size, cy), color, thick)
            cv2.line(canvas, (cx, cy), (cx, cy + dy * size), color, thick)

    # ── Skeleton drawing ───────────────────────────────────────────────────────

    def _draw_skeleton_on(
        self,
        img: np.ndarray,
        pose: HumanPose,
        letterbox_w: int,
        letterbox_h: int,
        orig_w: int,
        orig_h: int,
    ) -> np.ndarray:
        """
        Draw skeleton overlay. Landmarks are normalized to the original frame (orig_w × orig_h);
        the panel image is letterboxed to (letterbox_w × letterbox_h), so map into the
        inner content rectangle, not the full letterbox.
        """
        out = img.copy()
        ox, oy, nw, nh = self._letterbox_content_rect(
            orig_w, orig_h, letterbox_w, letterbox_h
        )

        def px_py(kp: Keypoint3D) -> tuple[int, int]:
            return (ox + int(kp.x * nw), oy + int(kp.y * nh))

        # Bones
        for a_name, b_name in SKELETON_PAIRS:
            a = pose.body.get(a_name)
            b = pose.body.get(b_name)
            if a and b and a.visibility > 0.4 and b.visibility > 0.4:
                pa, pb = px_py(a), px_py(b)
                cv2.line(out, pa, pb, C_SKELETON, 2)

        # Body joints
        for name, kp in pose.body.items():
            if kp.visibility > 0.5:
                cx, cy = px_py(kp)
                cv2.circle(out, (cx, cy), 5, (15, 20, 30), -1)
                cv2.circle(out, (cx, cy), 5, C_SKELETON, 1)

        # Hand landmarks (smaller)
        for kps in [pose.left_hand.values(), pose.right_hand.values()]:
            for kp in kps:
                cx, cy = px_py(kp)
                cv2.circle(out, (cx, cy), 2, C_HAND, -1)

        return out

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _letterbox_content_rect(
        orig_w: int, orig_h: int, target_w: int, target_h: int
    ) -> tuple[int, int, int, int]:
        """Match `_fit_frame`: offsets (ox,oy) and content size (nw,nh) inside target canvas."""
        if orig_w <= 0 or orig_h <= 0:
            return 0, 0, target_w, target_h
        scale = min(target_w / orig_w, target_h / orig_h)
        nw, nh = int(orig_w * scale), int(orig_h * scale)
        ox = (target_w - nw) // 2
        oy = (target_h - nh) // 2
        return ox, oy, nw, nh

    @staticmethod
    def _fit_frame(
        frame: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        """Resize frame maintaining aspect ratio with black letterbox."""
        if frame is None or frame.size == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        fh, fw = frame.shape[:2]
        scale = min(target_w / fw, target_h / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        resized = cv2.resize(frame, (nw, nh))

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        ox = (target_w - nw) // 2
        oy = (target_h - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        return canvas


def _extract_video_frames(
    video_path: str, n_frames: int
) -> list[np.ndarray]:
    """Extract n evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [np.zeros((480, 640, 3), dtype=np.uint8)] * n_frames

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max(n_frames, 1))
    frames, count = [], 0

    while len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % step == 0:
            frames.append(frame)

    cap.release()
    if frames and len(frames) < n_frames:
        frames += [frames[-1]] * (n_frames - len(frames))
    return frames[:n_frames]
