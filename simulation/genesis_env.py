"""
simulation/genesis_env.py — Genesis Simulator Environment
Fixes:
  - Auto-resolves H1 model path
  - Balance stabilizer prevents immediate falls
  - Returns human-readable render frames even in headless mode
"""
import os
import shutil
import numpy as np
from pathlib import Path
from typing import Optional

from config import config


def _get_genesis_assets_xml_dir() -> Optional[Path]:
    import genesis as gs

    p = Path(gs.__file__).parent / "assets" / "xml"
    return p if p.exists() else None


def _ensure_h1_model() -> str:
    genesis_xml_dir = _get_genesis_assets_xml_dir()

    if genesis_xml_dir:
        p = genesis_xml_dir / "unitree_h1" / "scene.xml"
        if p.exists():
            print(f"[Env] H1 found: {p}")
            return str(p)

    for cand in [
        Path("xml/unitree_h1/scene.xml"),
        Path("mujoco_menagerie/unitree_h1/scene.xml"),
        Path("assets/unitree_h1/scene.xml"),
    ]:
        abs_p = cand.resolve()
        if abs_p.exists():
            print(f"[Env] H1 found locally: {abs_p}")
            if genesis_xml_dir:
                _copy_to_genesis_assets(abs_p.parent, genesis_xml_dir)
            return str(abs_p)

    print("[Env] Downloading H1 via robot_descriptions...")
    from robot_descriptions import h1_mj_description

    scene = Path(h1_mj_description.MJCF_PATH)
    if scene.exists():
        if genesis_xml_dir:
            _copy_to_genesis_assets(scene.parent, genesis_xml_dir)
        print(f"[Env] robot_descriptions OK: {scene}")
        return str(scene)

    print("[Env] Trying git clone...")
    os.system(
        "git clone --depth 1 --filter=blob:none --sparse "
        "https://github.com/google-deepmind/mujoco_menagerie.git "
        "_tmp_menagerie -q && "
        "cd _tmp_menagerie && git sparse-checkout set unitree_h1 -q && cd .."
    )
    src = Path("_tmp_menagerie/unitree_h1")
    dest = Path("xml/unitree_h1")
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(str(dest))
        shutil.copytree(str(src), str(dest))
        shutil.rmtree("_tmp_menagerie", ignore_errors=True)
        scene = dest / "scene.xml"
        if scene.exists():
            abs_scene = scene.resolve()
            if genesis_xml_dir:
                _copy_to_genesis_assets(abs_scene.parent, genesis_xml_dir)
            return str(abs_scene)
    shutil.rmtree("_tmp_menagerie", ignore_errors=True)

    print("[Env] Using Genesis built-in humanoid fallback.")
    if genesis_xml_dir:
        fallback = genesis_xml_dir / "humanoid.xml"
        if fallback.exists():
            return str(fallback)

    raise RuntimeError("Cannot find Unitree H1. Run: pip install robot_descriptions")


def _copy_to_genesis_assets(src_dir: Path, genesis_xml_dir: Path):
    dest = genesis_xml_dir / "unitree_h1"
    if dest.exists():
        return
    shutil.copytree(str(src_dir), str(dest))
    print(f"[Env] Copied H1 → {dest}")


class HumanoidEnv:
    def __init__(self, show_viewer: bool = None, task: str = "default"):
        self.show_viewer = (
            show_viewer if show_viewer is not None
            else config.simulation.show_viewer
        )
        self.task = task
        self.dt = config.simulation.dt
        self.step_count = 0
        self.robot = None
        self.scene = None
        self._objects = {}
        self._render_camera = None
        self.metrics = {
            "distance_traveled": 0.0,
            "gait_stability": [],
            "fall_count": 0,
            "object_contacts": 0,
            "goal_reached": False,
        }
        self._init_simulator()

    def _init_simulator(self):
        import genesis as gs

        scene_path = _ensure_h1_model()
        print(f"[Env] Loading: {scene_path}")
        gs.init(backend=gs.cpu, logging_level="warning")

        vis_options = gs.options.VisOptions(
            ambient_light=(0.12, 0.12, 0.14),
        )
        raster = gs.renderers.Rasterizer()

        scene_kw = dict(
            vis_options=vis_options,
            show_viewer=self.show_viewer,
            sim_options=gs.options.SimOptions(dt=self.dt),
            renderer=raster,
        )
        if self.show_viewer:
            scene_kw["viewer_options"] = gs.options.ViewerOptions(
                res=(960, 720),
                camera_pos=(2.8, 0.0, 1.15),
                camera_lookat=(0.0, 0.0, 0.92),
                camera_fov=42,
                max_FPS=60,
            )

        self.scene = gs.Scene(**scene_kw)
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=scene_path))

        if config.simulation.offscreen_camera:
            w, h = config.simulation.camera_res
            px, py, pz = config.simulation.camera_pos
            lx, ly, lz = config.simulation.camera_lookat
            self._render_camera = self.scene.add_camera(
                res=(w, h),
                pos=(px, py, pz),
                lookat=(lx, ly, lz),
                fov=config.simulation.camera_fov,
                GUI=False,
            )
            print("[Env] Offscreen 3D camera ready (headless RGB)")

        self.scene.build()
        print("[Env] Genesis ready — Unitree H1 loaded ✓")

    def reset(self) -> dict:
        self.step_count = 0
        self.metrics = {
            "distance_traveled": 0.0,
            "gait_stability": [],
            "fall_count": 0,
            "object_contacts": 0,
            "goal_reached": False,
        }
        self.scene.reset()
        return self._get_obs()

    def step(self, joint_angles: dict) -> tuple:
        self.step_count += 1

        stabilized = self._apply_balance_stabilizer(joint_angles)
        self._apply_joints(stabilized)
        self.scene.step()

        obs = self._get_obs()
        return obs, self._reward(obs), self._done(obs), {}

    def execute_sequence(
        self,
        trajectory: list,
        render_frames: bool = False,
    ) -> list:
        """
        Execute trajectory and collect frames.
        Falls: tracked but stabilizer reduces them significantly.
        Frames: rendered from Genesis if available, else placeholder.
        """
        frames = []
        prev_com = np.zeros(3)

        for angles in trajectory:
            obs, _, done, _ = self.step(angles)
            com = np.array(obs.get("com_position", [0, 0, 0]))
            self.metrics["distance_traveled"] += float(
                np.linalg.norm(com - prev_com)
            )
            prev_com = com

            h = obs.get("base_height", 0.85)
            self.metrics["gait_stability"].append(1.0 if h > 0.35 else 0.0)

            if render_frames:
                frame = self._render_frame()
                frames.append(frame)

            if done:
                if obs.get("has_fallen", False):
                    self.metrics["fall_count"] += 1
                break

        s = self.metrics["gait_stability"]
        if isinstance(s, list):
            self.metrics["gait_stability"] = float(np.mean(s)) if s else 0.0

        return frames

    def add_object(self, object_type: str, position: tuple = (1.0, 0.0, 0.8)):
        import genesis as gs

        morph_map = {
            "cube":   gs.morphs.Box(size=(0.05, 0.05, 0.05)),
            "bottle": gs.morphs.Cylinder(radius=0.03, height=0.2),
            "bowl":   gs.morphs.Sphere(radius=0.08),
        }
        morph = morph_map.get(
            object_type, gs.morphs.Box(size=(0.05, 0.05, 0.05))
        )
        self._objects[object_type] = self.scene.add_entity(
            morph, pos=position
        )

    def get_metrics(self) -> dict:
        s = self.metrics["gait_stability"]
        if isinstance(s, list):
            s = float(np.mean(s)) if s else 0.0
        return {
            "distance_traveled": round(self.metrics["distance_traveled"], 3),
            "gait_stability": round(s, 3),
            "fall_count": self.metrics["fall_count"],
            "object_contacts": self.metrics["object_contacts"],
            "goal_reached": self.metrics["goal_reached"],
            "total_steps": self.step_count,
        }

    _STAND_POSE = {
        "left_hip_yaw": 0.0,  "left_hip_roll": 0.0,  "left_hip_pitch": 0.28,
        "left_knee": 0.62,    "left_ankle": -0.34,
        "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": 0.28,
        "right_knee": 0.62,   "right_ankle": -0.34,
        "torso": 0.0,
        "left_shoulder_pitch": 0.0,  "left_shoulder_roll": 0.15,
        "left_shoulder_yaw": 0.0,    "left_elbow": 0.4,   "left_wrist_roll": 0.0,
        "right_shoulder_pitch": 0.0, "right_shoulder_roll": -0.15,
        "right_shoulder_yaw": 0.0,   "right_elbow": 0.4,  "right_wrist_roll": 0.0,
    }

    def _apply_balance_stabilizer(
        self, angles: dict, alpha: float = 0.35
    ) -> dict:
        leg_joints = {
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
            "left_knee", "left_ankle",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
            "right_knee", "right_ankle", "torso",
        }

        stabilized = {}
        for joint, retargeted_val in angles.items():
            stand_val = self._STAND_POSE.get(joint, 0.0)
            a = alpha * 1.5 if joint in leg_joints else alpha * 0.5
            a = min(a, 1.0)
            stabilized[joint] = retargeted_val * (1 - a) + stand_val * a

        return stabilized

    def _apply_joints(self, angles: dict):
        if not self.robot:
            return
        all_j = (
            config.humanoid.locomotion_joints
            + config.humanoid.manipulation_joints
        )
        names = [k for k in angles if k in all_j]
        if not names:
            return
        self.robot.set_dofs_position(
            position=np.array([float(angles[n]) for n in names]),
            dofs_idx_local=[all_j.index(n) for n in names],
        )

    def _tensor_to_bgr(self, rgb) -> np.ndarray:
        """Genesis cameras return RGB; OpenCV / pipeline expect BGR uint8."""
        import cv2

        if hasattr(rgb, "detach"):
            rgb = rgb.detach()
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu()
        if hasattr(rgb, "numpy"):
            rgb = rgb.numpy()
        arr = np.asarray(rgb)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.5:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    def _render_frame(self) -> np.ndarray:
        if self._render_camera is not None:
            out = self._render_camera.render(rgb=True)
            if isinstance(out, (list, tuple)):
                rgb = out[0]
            else:
                rgb = out
            if rgb is not None:
                frame = self._tensor_to_bgr(rgb)
                if frame is not None and frame.size > 0:
                    return frame

        frame = self.scene.render()
        if frame is not None and getattr(frame, "size", 0) > 0:
            if hasattr(frame, "numpy"):
                frame = frame.numpy()
            return np.asarray(frame)
        return self._placeholder_frame()

    def _placeholder_frame(self) -> np.ndarray:
        import cv2

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 40
        h_val = self._get_obs().get("base_height", 0.85)
        s = self.metrics.get("gait_stability", [])
        stab = float(np.mean(s)) if isinstance(s, list) and s else 0.0

        cv2.putText(frame, "Unitree H1 Simulation",
                    (160, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        cv2.putText(frame, f"Step: {self.step_count}",
                    (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 200, 150), 1)
        cv2.putText(frame, f"Height: {h_val:.3f}m",
                    (220, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 200, 150), 1)
        cv2.putText(frame, f"Stability: {stab:.2f}",
                    (220, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 200, 150), 1)
        cv2.putText(frame, f"Task: {self.task[:40]}",
                    (80, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 100), 1)

        cx, cy = 320, 160
        cv2.circle(frame, (cx, cy - 60), 20, (100, 180, 255), 2)
        cv2.line(frame, (cx, cy - 40), (cx, cy + 40), (100, 180, 255), 2)
        cv2.line(frame, (cx, cy - 10), (cx - 40, cy + 20), (100, 180, 255), 2)
        cv2.line(frame, (cx, cy - 10), (cx + 40, cy + 20), (100, 180, 255), 2)
        cv2.line(frame, (cx, cy + 40), (cx - 20, cy + 90), (100, 180, 255), 2)
        cv2.line(frame, (cx, cy + 40), (cx + 20, cy + 90), (100, 180, 255), 2)

        return frame

    def _get_obs(self) -> dict:
        if not self.robot:
            return self._mock_obs()
        pos = self.robot.get_pos().numpy()
        h = float(pos[2]) if len(pos) > 2 else 0.98
        return {
            "base_position": pos.tolist(),
            "base_orientation": self.robot.get_quat().numpy().tolist(),
            "base_height": h,
            "com_position": pos.tolist(),
            "has_fallen": h < 0.3,
        }

    def _mock_obs(self) -> dict:
        return {
            "base_position": [0.0, 0.0, 0.98],
            "base_orientation": [0.0, 0.0, 0.0, 1.0],
            "base_height": 0.98,
            "com_position": [0.0, 0.0, 0.98],
            "has_fallen": False,
        }

    def _reward(self, obs: dict) -> float:
        h = float(obs.get("base_height", 0.0))
        return max(0.0, (h - 0.5) / 0.5)

    def _done(self, obs: dict) -> bool:
        return (
            obs.get("has_fallen", False)
            or self.step_count >= config.simulation.max_episode_steps
        )

    def close(self):
        if self.scene:
            self.scene.close()
