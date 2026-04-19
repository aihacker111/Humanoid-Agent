"""
config.py — System configuration
Project: LLM-Guided Kinematic Retargeting for Humanoid Robot Learning from Web Video
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


def _load_env():
    """Auto-load .env file from current dir, parents, or /content/ (Colab)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    search_dirs = (
        [Path.cwd()]
        + list(Path.cwd().parents)[:3]
        + [Path("/content"), Path(__file__).parent]
    )
    for d in search_dirs:
        env_file = d / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
            print(f"[Config] Loaded .env from: {env_file}")
            return


_load_env()


@dataclass
class OpenRouterConfig:
    api_key: str = os.getenv("OPENROUTER_API_KEY", "your-api-key-here")
    base_url: str = "https://openrouter.ai/api/v1"
    vision_model: str = "qwen/qwen3.6-plus"
    reasoning_model: str = "anthropic/claude-sonnet-4.6"
    fast_model: str = "anthropic/claude-sonnet-4.6"


@dataclass
class HumanoidConfig:
    model_name: str = "unitree_h1"
    """Label shown in comparison video / UI (simulation still loads MJCF from model_name)."""
    display_name: str = "Unitree H1"
    total_dof: int = 27
    locomotion_joints: list = field(default_factory=lambda: [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
        "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
        "right_knee", "right_ankle",
        "torso",
    ])
    manipulation_joints: list = field(default_factory=lambda: [
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
        "left_elbow", "left_wrist_roll",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
        "right_elbow", "right_wrist_roll",
    ])
    joint_limits: dict = field(default_factory=lambda: {
        "hip_pitch":      (-0.43, 0.43),
        "hip_roll":       (-0.43, 0.43),
        "hip_yaw":        (-0.43, 0.43),
        "knee":           (-0.26, 2.05),
        "ankle":          (-0.87, 0.52),
        "shoulder_pitch": (-3.14, 3.14),
        "shoulder_roll":  (-1.57, 1.57),
        "shoulder_yaw":   (-1.57, 1.57),
        "elbow":          (-1.57, 1.57),
        "wrist_roll":     (-1.57, 1.57),
        "torso":          (-2.35, 2.35),
    })


@dataclass
class PoseConfig:
    model_complexity: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    sample_rate: int = 5


@dataclass
class SimulationConfig:
    backend: str = "cpu"
    dt: float = 0.02
    gravity: tuple = (0, 0, -9.81)
    show_viewer: bool = False       # False for Colab; True = interactive 3D window (needs display)
    max_episode_steps: int = 500
    # Headless 3D RGB for videos: Genesis camera + Rasterizer (see simulation/genesis_env.py)
    offscreen_camera: bool = True
    camera_res: tuple = (640, 480)
    camera_pos: tuple = (2.8, 0.0, 1.15)
    camera_lookat: tuple = (0.0, 0.0, 0.92)
    camera_fov: float = 42.0


@dataclass
class SkillConfig:
    db_path: str = "outputs/skill_library.json"
    min_primitive_length: int = 10
    similarity_threshold: float = 0.85


@dataclass
class AppConfig:
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    humanoid: HumanoidConfig = field(default_factory=HumanoidConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    skill: SkillConfig = field(default_factory=SkillConfig)
    debug: bool = False
    output_dir: str = "outputs"


config = AppConfig()
