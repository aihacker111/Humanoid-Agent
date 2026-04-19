"""
models/__init__.py — Pydantic schemas for the entire pipeline
"""
from __future__ import annotations
from pydantic import BaseModel
from typing import Optional
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"; MEDIUM = "MEDIUM"; HIGH = "HIGH"; CRITICAL = "CRITICAL"


class ActionType(str, Enum):
    WALK_FORWARD   = "walk_forward"
    WALK_BACKWARD  = "walk_backward"
    TURN_LEFT      = "turn_left"
    TURN_RIGHT     = "turn_right"
    STAND          = "stand"
    SIT            = "sit"
    CROUCH         = "crouch"
    STEP_OVER      = "step_over"
    REACH          = "reach"
    GRASP          = "grasp"
    PLACE          = "place"
    POUR           = "pour"
    PUSH           = "push"
    PULL           = "pull"
    ROTATE_OBJECT  = "rotate_object"
    BIMANUAL       = "bimanual"
    LOCO_MANIP     = "loco_manip"
    UNKNOWN        = "unknown"


class ControlMode(str, Enum):
    LOCOMOTION   = "locomotion"
    MANIPULATION = "manipulation"
    WHOLE_BODY   = "whole_body"


class Keypoint3D(BaseModel):
    x: float; y: float; z: float
    visibility: float = 1.0


class HumanPose(BaseModel):
    frame_number: int
    timestamp: float
    body: dict[str, Keypoint3D] = {}
    left_hand: dict[str, Keypoint3D] = {}
    right_hand: dict[str, Keypoint3D] = {}
    is_moving: bool = False
    center_of_mass: Optional[Keypoint3D] = None
    dominant_mode: ControlMode = ControlMode.LOCOMOTION


class MotionPrimitive(BaseModel):
    primitive_id: str
    action_type: ActionType
    control_mode: ControlMode
    start_frame: int
    end_frame: int
    duration_seconds: float
    joint_trajectory: list[dict[str, float]] = []
    velocity_profile: list[float] = []
    description: str = ""
    reasoning: str = ""
    object_interactions: list[str] = []
    confidence: float = 0.0
    is_valid: bool = True


class SkillSequence(BaseModel):
    task_description: str
    primitives: list[MotionPrimitive] = []
    estimated_duration: float = 0.0
    requires_locomotion: bool = False
    requires_manipulation: bool = False


class RetargetedPose(BaseModel):
    frame_number: int
    source_human_pose: HumanPose
    joint_angles: dict[str, float] = {}
    retargeting_error: float = 0.0
    balance_score: float = 0.0
    reachability_score: float = 0.0
    retargeting_reasoning: str = ""
    constraint_violations: list[str] = []


class RetargetedSequence(BaseModel):
    primitive_id: str
    frames: list[RetargetedPose] = []
    success: bool = True
    failure_reason: Optional[str] = None


class SimulationResult(BaseModel):
    task_description: str
    skill_sequence_id: str
    task_success: bool = False
    success_rate: float = 0.0
    distance_traveled: float = 0.0
    gait_stability: float = 0.0
    fall_count: int = 0
    object_contact_success: float = 0.0
    goal_reached: bool = False
    grasp_success: float = 0.0
    motion_naturalness: float = 0.0
    coordination_score: float = 0.0
    execution_time_seconds: float = 0.0
    total_frames: int = 0
    model_used: str = ""
    retargeting_method: str = "llm_guided"


class SceneContext(BaseModel):
    frame_number: int
    description: str = ""
    detected_objects: list[str] = []
    human_activity: str = ""
    locomotion_observed: bool = False
    manipulation_observed: bool = False
    environment_type: str = ""
    task_goal: Optional[str] = None


class TaskPlan(BaseModel):
    task: str
    subtasks: list[str] = []
    required_skills: list[str] = []
    estimated_steps: int = 0
    requires_locomotion: bool = False
    requires_manipulation: bool = False
    reasoning: str = ""
