"""
core/side_by_side_renderer.py — Professional comparison video renderer
Robot panel: REAL Genesis 3D frame only. No 2D drawing.
"""
import cv2, numpy as np
from pathlib import Path
from typing import Optional
from models import HumanPose, RetargetedPose
from config import config

W=1280; H=720; PANEL_W=620; PANEL_H=580; HEADER_H=60; FOOTER_H=80; DIVIDER_W=40
BG=(18,18,24); PANEL_BG=(26,28,38); HEADER_BG=(12,14,20); DIVIDER_BG=(30,32,45)
ACCENT_BLUE=(64,156,255); ACCENT_TEAL=(80,220,180); ACCENT_GOLD=(60,200,255)
TEXT_PRI=(230,235,245); TEXT_SEC=(130,140,160); TEXT_DIM=(70,80,100)
SUCCESS=(80,200,120); WARNING=(80,180,255); DANGER=(80,80,220)
FONT=cv2.FONT_HERSHEY_SIMPLEX
BONES=[
    ("left_shoulder","right_shoulder"),
    ("left_shoulder","left_elbow"),("left_elbow","left_wrist"),
    ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
    ("left_shoulder","left_hip"),("right_shoulder","right_hip"),
    ("left_hip","right_hip"),
    ("left_hip","left_knee"),("left_knee","left_ankle"),
    ("right_hip","right_knee"),("right_knee","right_ankle"),
]


class SideBySideRenderer:
    def __init__(self, output_path, task="", fps=15.0,
                 paper_title="LLM-Guided Kinematic Retargeting"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.task = task; self.fps = fps; self.paper_title = paper_title
        self.frame_count = 0
        self._nat = 0.0; self._bal = 0.0; self._coord = 0.0; self._falls = 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (W, H))
        print(f"[Renderer] Output: {self.output_path}  ({W}×{H} @ {fps}fps)")

    def write_frame(self, human_frame=None, robot_frame=None,
                    human_pose=None, retargeted_pose=None, metrics=None):
        if metrics:
            self._nat   = float(metrics.get("naturalness",   self._nat))
            self._bal   = float(metrics.get("balance",       self._bal))
            self._coord = float(metrics.get("coordination",  self._coord))
            self._falls = int(metrics.get("fall_count",      self._falls))

        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        self._header(canvas)
        canvas[HEADER_H:HEADER_H+PANEL_H, 0:PANEL_W] = \
            self._human_panel(human_frame, human_pose)
        self._divider(canvas)
        canvas[HEADER_H:HEADER_H+PANEL_H, PANEL_W+DIVIDER_W:PANEL_W+DIVIDER_W+PANEL_W] = \
            self._robot_panel(robot_frame, retargeted_pose)
        self._footer(canvas)
        self.writer.write(canvas)
        self.frame_count += 1

    def write_batch(self, human_frames, robot_frames,
                    poses=None, retargeted=None, metrics_list=None):
        n = max(len(human_frames), len(robot_frames) if robot_frames else 0)
        for i in range(n):
            hf = human_frames[i] if i < len(human_frames) else None
            rf = robot_frames[i] if robot_frames and i < len(robot_frames) else None
            self.write_frame(
                human_frame=hf, robot_frame=rf,
                human_pose=poses[i] if poses and i < len(poses) else None,
                retargeted_pose=retargeted[i] if retargeted and i < len(retargeted) else None,
                metrics=metrics_list[i] if metrics_list and i < len(metrics_list) else None,
            )

    def add_title_card(self, title, subtitle="", duration_seconds=2.0):
        n = int(duration_seconds * self.fps)
        for _ in range(n):
            c = np.full((H, W, 3), BG, dtype=np.uint8)
            my = H // 2
            cv2.line(c, (120, my-60), (W-120, my-60), DIVIDER_BG, 1)
            cv2.line(c, (120, my+60), (W-120, my+60), DIVIDER_BG, 1)
            for x in [120, W//2, W-120]:
                cv2.circle(c, (x, my-60), 3, ACCENT_BLUE, -1)
                cv2.circle(c, (x, my+60), 3, ACCENT_TEAL, -1)
            self._t(c, self.paper_title.upper(), W//2, my-90, TEXT_DIM, 0.42, cx=True)
            self._t(c, title, W//2, my-18, TEXT_PRI, 1.0, th=2, cx=True)
            if subtitle:
                self._t(c, subtitle, W//2, my+28, TEXT_SEC, 0.52, cx=True)
            model = config.openrouter.reasoning_model.split("/")[-1]
            self._t(c, f"Model: {model}", W//2, my+86, TEXT_DIM, 0.38, cx=True)
            self.writer.write(c)
            self.frame_count += 1

    def finalize(self) -> str:
        self.writer.release()
        mb = self.output_path.stat().st_size / 1e6
        print(f"[Renderer] ✓ {self.frame_count} frames → {self.output_path.name} ({mb:.1f}MB)")
        return str(self.output_path)

    # ── Panels ─────────────────────────────────────────────────────────────────

    def _human_panel(self, frame, pose) -> np.ndarray:
        p = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
        cv2.rectangle(p, (0,0), (PANEL_W,36), (20,22,35), -1)
        cv2.rectangle(p, (0,0), (4,36), ACCENT_BLUE, -1)
        self._t(p, "HUMAN DEMONSTRATION", 14, 23, ACCENT_BLUE, 0.5)
        if pose:
            mode = pose.dominant_mode.value.upper()
            mc = {"LOCOMOTION":(255,140,50),"MANIPULATION":ACCENT_BLUE,
                  "WHOLE_BODY":ACCENT_TEAL}.get(mode, TEXT_SEC)
            bw = len(mode)*8+16; bx = PANEL_W-bw-8
            cv2.rectangle(p, (bx,6), (PANEL_W-8,30), mc, 1)
            self._t(p, mode, bx+8, 23, mc, 0.38)

        VY=36; VH=PANEL_H-36-36
        if frame is not None and frame.size > 0:
            fh,fw = frame.shape[:2]
            scale = min(PANEL_W/fw, VH/fh)
            nw,nh = int(fw*scale), int(fh*scale)
            resized = cv2.resize(frame, (nw,nh))
            ox=(PANEL_W-nw)//2; oy=VY+(VH-nh)//2
            p[oy:oy+nh, ox:ox+nw] = resized
            if pose:
                self._skeleton(p, pose, ox, oy, nw, nh)
        else:
            self._t(p, "No video frame", PANEL_W//2, VY+VH//2, TEXT_DIM, 0.5, cx=True)

        iy = PANEL_H-36
        cv2.rectangle(p, (0,iy), (PANEL_W,PANEL_H), (20,22,35), -1)
        if pose:
            lh = len(pose.left_hand) > 0; rh = len(pose.right_hand) > 0
            self._t(p, f"Body:{len(pose.body)}kpts  L.Hand:{lh}  R.Hand:{rh}  Moving:{pose.is_moving}",
                    10, iy+22, TEXT_SEC, 0.36)
        return p

    def _robot_panel(self, frame, retargeted) -> np.ndarray:
        """
        Robot panel shows ONLY the real Genesis 3D render.
        If frame is None (Genesis not available), shows a waiting message.
        No 2D stick figures, no fake drawings.
        """
        p = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)

        # Header
        cv2.rectangle(p, (0,0), (PANEL_W,36), (20,35,30), -1)
        cv2.rectangle(p, (0,0), (4,36), ACCENT_TEAL, -1)
        self._t(p, "UNITREE H1  —  GENESIS 3D SIM", 14, 23, ACCENT_TEAL, 0.5)
        if retargeted:
            b = retargeted.balance_score
            bc = SUCCESS if b>0.7 else WARNING if b>0.4 else DANGER
            badge = f"BAL {b:.0%}"
            bx = PANEL_W - len(badge)*8 - 24
            cv2.rectangle(p, (bx,6), (PANEL_W-8,30), bc, 1)
            self._t(p, badge, bx+8, 23, bc, 0.38)

        VY=36; VH=PANEL_H-36-44

        # ── Show ONLY real Genesis 3D frame ────────────────────────────────
        has_frame = (
            frame is not None
            and hasattr(frame, 'shape')
            and frame.size > 0
            and frame.mean() > 5   # not fully black
        )

        if has_frame:
            fh,fw = frame.shape[:2]
            scale = min(PANEL_W/fw, VH/fh)
            nw,nh = int(fw*scale), int(fh*scale)
            resized = cv2.resize(frame, (nw,nh), interpolation=cv2.INTER_LINEAR)
            ox=(PANEL_W-nw)//2; oy=VY+(VH-nh)//2
            p[oy:oy+nh, ox:ox+nw] = resized

            # Compact joint readout overlay on bottom-left of frame
            if retargeted:
                self._joint_overlay(p, retargeted, ox+6, oy+nh-80)
        else:
            # Genesis not available — simple waiting message, no fake drawing
            cy = VY + VH//2
            cv2.rectangle(p, (40, VY+40), (PANEL_W-40, VY+VH-40), DIVIDER_BG, 1)
            self._t(p, "Genesis 3D", PANEL_W//2, cy-20, TEXT_SEC, 0.7, cx=True)
            self._t(p, "render unavailable", PANEL_W//2, cy+8, TEXT_DIM, 0.42, cx=True)
            self._t(p, "(Genesis not installed or CPU render pending)",
                    PANEL_W//2, cy+32, TEXT_DIM, 0.32, cx=True)

        # Bottom info
        iy = PANEL_H-36
        cv2.rectangle(p, (0,iy), (PANEL_W,PANEL_H), (20,35,30), -1)
        source = "Genesis 3D" if has_frame else "awaiting"
        if retargeted:
            self._t(p, f"Reach:{retargeted.reachability_score:.2f}  "
                       f"DOF:{len(retargeted.joint_angles)}  [{source}]",
                    10, iy+22, TEXT_SEC, 0.38)
        return p

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _skeleton(self, panel, pose, ox, oy, nw, nh):
        def px(kp):
            x = max(ox, min(ox+nw-1, ox + int(kp.x * nw)))
            y = max(oy, min(oy+nh-1, oy + int(kp.y * nh)))
            return (x, y)
        for a, b in BONES:
            ka, kb = pose.body.get(a), pose.body.get(b)
            if ka and kb and ka.visibility > 0.4 and kb.visibility > 0.4:
                cv2.line(panel, px(ka), px(kb), (60,210,90), 2, cv2.LINE_AA)
        for name, kp in pose.body.items():
            if kp.visibility > 0.5:
                pt = px(kp)
                key = any(j in name for j in
                          ["shoulder","elbow","wrist","hip","knee","ankle"])
                r = 6 if key else 3
                cv2.circle(panel, pt, r, (100,255,120) if key else (60,170,80), -1, cv2.LINE_AA)
                cv2.circle(panel, pt, r, (220,255,225), 1, cv2.LINE_AA)
        for kps in [pose.left_hand.values(), pose.right_hand.values()]:
            for kp in kps:
                cv2.circle(panel, px(kp), 3, ACCENT_GOLD, -1, cv2.LINE_AA)

    def _joint_overlay(self, panel, retargeted, x, y):
        """4 key joint angles overlaid on the Genesis frame."""
        angles = retargeted.joint_angles
        items = [
            ("L.Elbow",  "left_elbow"),
            ("R.Elbow",  "right_elbow"),
            ("L.Knee",   "left_knee"),
            ("R.Knee",   "right_knee"),
        ]
        ov = panel.copy()
        cv2.rectangle(ov, (x-3, y-14), (x+130, y+len(items)*15+2), (0,0,0), -1)
        cv2.addWeighted(ov, 0.55, panel, 0.45, 0, panel)
        for i, (label, joint) in enumerate(items):
            val = angles.get(joint, 0.0)
            deg = round(val * 57.3, 1)
            color = ACCENT_GOLD if abs(val) > 0.3 else TEXT_SEC
            self._t(panel, f"{label}: {deg:+.1f}°", x, y+i*15, color, 0.34)

    def _header(self, c):
        cv2.rectangle(c, (0,0), (W,HEADER_H), HEADER_BG, -1)
        cv2.line(c, (0,HEADER_H-1), (W,HEADER_H-1), DIVIDER_BG, 1)
        self._t(c, self.paper_title.upper(), 16, 20, TEXT_DIM, 0.42)
        task = self.task[:70]+"..." if len(self.task)>70 else self.task
        self._t(c, f"Task: {task}", W//2, 20, ACCENT_GOLD, 0.44, cx=True)
        model = config.openrouter.reasoning_model.split("/")[-1]
        self._t(c, f"Frame {self.frame_count:04d}  |  {model}", W-16, 20, TEXT_DIM, 0.38, rt=True)
        self._t(c, "Human Demo", 16, 44, ACCENT_BLUE, 0.42)
        self._t(c, "vs", W//2, 44, TEXT_DIM, 0.4, cx=True)
        self._t(c, "Unitree H1 (Genesis 3D)", W-16, 44, ACCENT_TEAL, 0.42, rt=True)

    def _footer(self, c):
        y0 = HEADER_H + PANEL_H
        cv2.rectangle(c, (0,y0), (W,H), HEADER_BG, -1)
        cv2.line(c, (0,y0), (W,y0), DIVIDER_BG, 1)
        items = [("Naturalness",self._nat,ACCENT_BLUE),
                 ("Balance",    self._bal,ACCENT_TEAL),
                 ("Coordination",self._coord,ACCENT_GOLD)]
        bw = (W-80)//3-20
        for i,(label,val,color) in enumerate(items):
            x = 40+i*(bw+20); my = y0+22
            self._t(c, label.upper(), x, my, TEXT_DIM, 0.36)
            self._t(c, f"{val:.0%}", x+bw-4, my, color, 0.44, rt=True)
            by = my+8
            cv2.rectangle(c, (x,by), (x+bw,by+10), (35,38,50), -1)
            cv2.rectangle(c, (x,by), (x+bw,by+10), (50,55,70), 1)
            fw2 = int(bw*np.clip(val,0,1))
            if fw2 > 0: cv2.rectangle(c, (x,by), (x+fw2,by+10), color, -1)
        fc = DANGER if self._falls > 0 else SUCCESS
        self._t(c, f"Falls: {self._falls}", W-40, y0+50, fc, 0.45, rt=True)
        self._t(c, f"Step {self.frame_count:04d}", 40, y0+50, TEXT_DIM, 0.38)

    def _divider(self, c):
        x0=PANEL_W; y0=HEADER_H; y1=y0+PANEL_H
        cv2.rectangle(c, (x0,y0), (x0+DIVIDER_W,y1), DIVIDER_BG, -1)
        mx=x0+DIVIDER_W//2; my=y0+PANEL_H//2
        self._t(c, "VS", mx, my-8, TEXT_SEC, 0.5, cx=True)
        for dy in [-40,-20,20,40]:
            cv2.circle(c, (mx,my+dy), 2, TEXT_DIM, -1)

    @staticmethod
    def _t(img, text, x, y, color, scale, th=1, cx=False, rt=False):
        (tw,_),_ = cv2.getTextSize(text, FONT, scale, th)
        if cx: x -= tw//2
        elif rt: x -= tw
        cv2.putText(img, text, (x,y), FONT, scale, color, th, cv2.LINE_AA)


def create_comparison_video(session_id, source_video, robot_frames,
                             poses, retargeted_frames, task, metrics_list=None):
    out = f"outputs/videos/{session_id}_comparison.mp4"
    renderer = SideBySideRenderer(out, task=task)
    renderer.add_title_card("Human Demo  vs  Unitree H1",
                             f"Task: {task}", 2.0)
    hf = _extract_frames(source_video, max(len(robot_frames), len(retargeted_frames)))
    renderer.write_batch(human_frames=hf, robot_frames=robot_frames,
                         poses=poses, retargeted=retargeted_frames,
                         metrics_list=metrics_list)
    return renderer.finalize()


def _extract_frames(video_path, n):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [None] * n
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max(n, 1))
    frames = []; count = 0
    while len(frames) < n:
        ret, f = cap.read()
        if not ret: break
        count += 1
        if count % step == 0: frames.append(f)
    cap.release()
    if frames and len(frames) < n:
        frames += [frames[-1]] * (n - len(frames))
    return frames[:n] if frames else [None] * n
