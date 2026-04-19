"""
simulation/genesis_env.py
Real 3D rendering via Genesis camera — no placeholder, no 2D drawing.
"""
import os, shutil, numpy as np, cv2
from pathlib import Path
from typing import Optional
from config import config


def _get_genesis_xml_dir():
    try:
        import genesis as gs
        p = Path(gs.__file__).parent / "assets" / "xml"
        return p if p.exists() else None
    except: return None


def _ensure_h1_model() -> str:
    gd = _get_genesis_xml_dir()
    if gd:
        p = gd / "unitree_h1" / "scene.xml"
        if p.exists(): return str(p)
    for c in [Path("xml/unitree_h1/scene.xml"), Path("mujoco_menagerie/unitree_h1/scene.xml")]:
        a = c.resolve()
        if a.exists():
            if gd: _copy_to_genesis(a.parent, gd)
            return str(a)
    print("[Env] Downloading H1...")
    try:
        try: import robot_descriptions
        except: os.system("pip install robot_descriptions -q")
        from robot_descriptions import h1_mj_description
        s = Path(h1_mj_description.MJCF_PATH)
        if s.exists():
            if gd: _copy_to_genesis(s.parent, gd)
            return str(s)
    except Exception as e: print(f"[Env] robot_descriptions: {e}")
    try:
        os.system("git clone --depth 1 --filter=blob:none --sparse "
                  "https://github.com/google-deepmind/mujoco_menagerie.git _tmp -q && "
                  "cd _tmp && git sparse-checkout set unitree_h1 -q && cd ..")
        src = Path("_tmp/unitree_h1"); dst = Path("xml/unitree_h1")
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists(): shutil.rmtree(str(dst))
            shutil.copytree(str(src), str(dst))
            shutil.rmtree("_tmp", ignore_errors=True)
            s = dst / "scene.xml"
            if s.exists():
                a = s.resolve()
                if gd: _copy_to_genesis(a.parent, gd)
                return str(a)
        shutil.rmtree("_tmp", ignore_errors=True)
    except Exception as e: print(f"[Env] git clone: {e}")
    if gd:
        fb = gd / "humanoid.xml"
        if fb.exists(): return str(fb)
    raise RuntimeError("Cannot find H1. Run: !pip install robot_descriptions")


def _copy_to_genesis(src, gd):
    dst = gd / "unitree_h1"
    if dst.exists(): return
    try: shutil.copytree(str(src), str(dst))
    except: pass


class HumanoidEnv:

    _STAND = {
        "left_hip_yaw":0.0,"left_hip_roll":0.0,"left_hip_pitch":0.28,
        "left_knee":0.62,"left_ankle":-0.34,
        "right_hip_yaw":0.0,"right_hip_roll":0.0,"right_hip_pitch":0.28,
        "right_knee":0.62,"right_ankle":-0.34,"torso":0.0,
        "left_shoulder_pitch":0.0,"left_shoulder_roll":0.15,"left_shoulder_yaw":0.0,
        "left_elbow":0.3,"left_wrist_roll":0.0,
        "right_shoulder_pitch":0.0,"right_shoulder_roll":-0.15,"right_shoulder_yaw":0.0,
        "right_elbow":0.3,"right_wrist_roll":0.0,
    }

    def __init__(self, show_viewer=None, task="default"):
        self.show_viewer = show_viewer if show_viewer is not None else config.simulation.show_viewer
        self.task = task
        self.dt = config.simulation.dt
        self.step_count = 0
        self.robot = None
        self.scene = None
        self.cam = None
        self._mock = False
        self.metrics = {"distance_traveled":0.0,"gait_stability":[],"fall_count":0,"object_contacts":0,"goal_reached":False}
        self._init()

    def _init(self):
        try:
            import genesis as gs
            path = _ensure_h1_model()
            print(f"[Env] Loading H1: {path}")
            gs.init(backend=gs.cpu, logging_level="warning")
            self.scene = gs.Scene(
                show_viewer=self.show_viewer,
                sim_options=gs.options.SimOptions(dt=self.dt),
                vis_options=gs.options.VisOptions(
                    show_world_frame=False,
                    ambient_light=(0.5, 0.5, 0.5),
                ),
                renderer=gs.renderers.Rasterizer(),
            )
            self.scene.add_entity(gs.morphs.Plane())
            self.robot = self.scene.add_entity(gs.morphs.MJCF(file=path))

            # ── 3D camera — GUI=False works in Colab headless ──────────────
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0.0, 0.0, 1.0),
                fov=40,
                GUI=False,
            )
            self.scene.build()
            print("[Env] Genesis ready — H1 + 3D camera ✓")
        except ImportError:
            print("[Env] Genesis not installed — MOCK")
            self._mock = True
        except Exception as e:
            print(f"[Env] Init error: {e}")
            self._mock = True

    def render_frame(self, orbit_deg: float = 0.0) -> Optional[np.ndarray]:
        """
        Render real 3D frame from Genesis camera.
        Returns BGR numpy array (H,W,3) or None if Genesis unavailable.
        """
        if self._mock or self.cam is None:
            return None   # caller handles None — no fake frame

        try:
            if abs(orbit_deg) > 0.01:
                r = 3.5
                a = np.radians(orbit_deg)
                self.cam.set_pose(
                    pos=(r * np.sin(a), r * np.cos(a), 2.5),
                    lookat=(0.0, 0.0, 1.0),
                )

            out = self.cam.render()
            rgb = out[0] if isinstance(out, tuple) else out   # (H,W,3) uint8 RGB

            if rgb is not None and rgb.size > 0:
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        except Exception as e:
            if config.debug:
                print(f"[Env] render_frame error: {e}")

        return None

    def reset(self):
        self.step_count = 0
        self.metrics = {"distance_traveled":0.0,"gait_stability":[],"fall_count":0,"object_contacts":0,"goal_reached":False}
        if not self._mock:
            try: self.scene.reset()
            except: pass
        return self._obs()

    def step(self, angles: dict):
        self.step_count += 1
        if self._mock:
            return self._obs(), 0.35, self.step_count >= config.simulation.max_episode_steps, {}
        self._apply(self._stabilize(angles))
        self.scene.step()
        obs = self._obs()
        return obs, self._reward(obs), self._done(obs), {}

    def execute_sequence(self, trajectory: list, render_frames=False):
        frames = []
        prev = np.zeros(3)
        for i, angles in enumerate(trajectory):
            obs, _, done, _ = self.step(angles)
            com = np.array(obs.get("com_position", [0,0,0]))
            self.metrics["distance_traveled"] += float(np.linalg.norm(com - prev))
            prev = com
            self.metrics["gait_stability"].append(1.0 if obs.get("base_height",0) > 0.5 else 0.0)
            if render_frames:
                orbit = (i / max(len(trajectory)-1, 1)) * 30
                frames.append(self.render_frame(orbit_deg=orbit))
            if done:
                if obs.get("has_fallen"): self.metrics["fall_count"] += 1
                break
        s = self.metrics["gait_stability"]
        self.metrics["gait_stability"] = float(np.mean(s)) if s else 0.0
        return frames

    def get_metrics(self):
        s = self.metrics["gait_stability"]
        if isinstance(s, list): s = float(np.mean(s)) if s else 0.0
        return {"distance_traveled":round(self.metrics["distance_traveled"],3),
                "gait_stability":round(s,3),"fall_count":self.metrics["fall_count"],
                "object_contacts":self.metrics["object_contacts"],
                "goal_reached":self.metrics["goal_reached"],"total_steps":self.step_count}

    def add_object(self, kind, pos=(1.0,0.0,0.8)):
        if self._mock: return
        try:
            import genesis as gs
            m = {"cube":gs.morphs.Box(size=(0.05,0.05,0.05)),
                 "bottle":gs.morphs.Cylinder(radius=0.03,height=0.2),
                 "bowl":gs.morphs.Sphere(radius=0.08)}.get(kind, gs.morphs.Box(size=(0.05,0.05,0.05)))
            self.scene.add_entity(m, pos=pos)
        except Exception as e: print(f"[Env] add_object: {e}")

    def _stabilize(self, angles, alpha=0.3):
        legs = {"left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle",
                "right_hip_yaw","right_hip_roll","right_hip_pitch","right_knee","right_ankle","torso"}
        return {j: float(v*(1-(alpha*1.4 if j in legs else alpha*0.4)) +
                         self._STAND.get(j,0.0)*(alpha*1.4 if j in legs else alpha*0.4))
                for j,v in angles.items()}

    def _apply(self, angles):
        if not self.robot: return
        try:
            all_j = config.humanoid.locomotion_joints + config.humanoid.manipulation_joints
            names = [k for k in angles if k in all_j]
            if names:
                self.robot.set_dofs_position(
                    position=np.array([float(angles[n]) for n in names]),
                    dofs_idx_local=[all_j.index(n) for n in names])
        except Exception as e:
            if config.debug: print(f"[Env] apply joints: {e}")

    def _obs(self):
        if self._mock or not self.robot:
            return {"base_position":[0,0,0.98],"base_orientation":[0,0,0,1],
                    "base_height":0.98,"com_position":[0,0,0.98],"has_fallen":False}
        try:
            pos = self.robot.get_pos().numpy()
            h = float(pos[2]) if len(pos) > 2 else 0.98
            return {"base_position":pos.tolist(),"base_orientation":self.robot.get_quat().numpy().tolist(),
                    "base_height":h,"com_position":pos.tolist(),"has_fallen":h < 0.3}
        except: return {"base_position":[0,0,0.98],"base_orientation":[0,0,0,1],
                        "base_height":0.98,"com_position":[0,0,0.98],"has_fallen":False}

    def _reward(self, obs): return max(0.0,(float(obs.get("base_height",0))-0.5)/0.5)
    def _done(self, obs): return obs.get("has_fallen",False) or self.step_count >= config.simulation.max_episode_steps
    def close(self):
        if self.scene and not self._mock:
            try: self.scene.close()
            except: pass
