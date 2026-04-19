"""
agents/skill_agent.py — Action Segmentation & Skill Library
Fixed: LLM-returned action types are validated and mapped safely
"""
import json
import uuid
import numpy as np
from pathlib import Path

from models import (
    HumanPose, MotionPrimitive, SkillSequence,
    ActionType, ControlMode, SceneContext,
)
from core.openrouter import OpenRouterClient, _extract_json_safe
from config import config


# All valid action type values — used to validate/map LLM output
VALID_ACTION_TYPES = {a.value for a in ActionType}

# Map common LLM variants → valid ActionType values
ACTION_TYPE_MAP = {
    "stand_manipulate":  "bimanual",
    "standing":          "stand",
    "walking":           "walk_forward",
    "pick":              "grasp",
    "pickup":            "grasp",
    "put_down":          "place",
    "wave":              "reach",
    "clap":              "bimanual",
    "rub_hands":         "bimanual",
    "chalk_application": "bimanual",
    "hand_preparation":  "bimanual",
    "bilateral_motion":  "bimanual",
}


def _safe_action_type(raw: str) -> ActionType:
    """Convert LLM string → valid ActionType, never raise."""
    if not raw:
        return ActionType.UNKNOWN
    val = raw.lower().strip().replace(" ", "_").replace("-", "_")
    if val in VALID_ACTION_TYPES:
        return ActionType(val)
    if val in ACTION_TYPE_MAP:
        return ActionType(ACTION_TYPE_MAP[val])
    # Partial match
    for valid in VALID_ACTION_TYPES:
        if val in valid or valid in val:
            return ActionType(valid)
    return ActionType.UNKNOWN


def _safe_control_mode(raw: str) -> ControlMode:
    """Convert LLM string → valid ControlMode, never raise."""
    if not raw:
        return ControlMode.MANIPULATION
    val = raw.lower().strip().replace(" ", "_").replace("-", "_")
    mode_map = {
        "locomotion":    ControlMode.LOCOMOTION,
        "manipulation":  ControlMode.MANIPULATION,
        "whole_body":    ControlMode.WHOLE_BODY,
        "whole body":    ControlMode.WHOLE_BODY,
        "loco_manip":    ControlMode.WHOLE_BODY,
        "both":          ControlMode.WHOLE_BODY,
    }
    return mode_map.get(val, ControlMode.MANIPULATION)


SEGMENTATION_PROMPT = """
You are an expert in human motion analysis for robot learning.

Analyze this video frame and pose sequence to identify motion primitives.

Scene: {scene_description}
Pose summary: {pose_summary}
Duration: {duration:.1f} seconds, {frame_count} frames

Return ONLY valid JSON, no extra text:
{{
  "primitives": [
    {{
      "action_type": "walk_forward|walk_backward|turn_left|turn_right|stand|sit|crouch|reach|grasp|place|pour|push|pull|rotate_object|bimanual|loco_manip|unknown",
      "control_mode": "locomotion|manipulation|whole_body",
      "start_frame_idx": 0,
      "end_frame_idx": 10,
      "description": "brief description",
      "reasoning": "why this is this action type",
      "object_interactions": [],
      "confidence": 0.8
    }}
  ],
  "task_goal": "overall task goal"
}}

IMPORTANT: action_type must be exactly one of the listed values.
"""

SKILL_COMPOSE_PROMPT = """
You are a robot task planning expert.

Available skills:
{skill_library}

Task: "{task}"

Return ONLY valid JSON:
{{
  "subtasks": ["step 1", "step 2"],
  "skill_ids": ["id1", "id2"],
  "requires_locomotion": false,
  "requires_manipulation": true,
  "reasoning": "explanation",
  "estimated_steps": 50
}}
"""


class SkillExtractionAgent:
    def __init__(self):
        self.client = OpenRouterClient()
        self.model = config.openrouter.vision_model
        self.reasoning_model = config.openrouter.reasoning_model
        self.skill_library: dict[str, MotionPrimitive] = {}
        self._load_library()

    def segment_video(
        self,
        poses: list[HumanPose],
        scene: SceneContext,
        frames: list = None,
    ) -> list[MotionPrimitive]:
        if not poses:
            return []

        pose_summary = self._summarize_poses(poses)
        duration = poses[-1].timestamp - poses[0].timestamp

        prompt = SEGMENTATION_PROMPT.format(
            scene_description=scene.description or "Unknown scene",
            pose_summary=json.dumps(pose_summary, indent=2),
            duration=duration,
            frame_count=len(poses),
        )

        try:
            if frames and len(frames) > 0:
                mid_frame = frames[len(frames) // 2]
                response = self.client.call_vision(
                    prompt=prompt, frame=mid_frame, model=self.model
                )
            else:
                response = self.client.call_text(
                    system="You are an expert in human motion analysis.",
                    user=prompt,
                    model=self.model,
                )

            data = self.client.extract_json(response)
            primitives = []

            for p_data in data.get("primitives", []):
                start_idx = min(
                    int(p_data.get("start_frame_idx", 0)), len(poses) - 1
                )
                end_idx = min(
                    int(p_data.get("end_frame_idx", len(poses) - 1)),
                    len(poses) - 1,
                )
                start_time = poses[start_idx].timestamp
                end_time = poses[end_idx].timestamp

                # Safe enum conversion — never crash on unknown values
                action_type = _safe_action_type(
                    p_data.get("action_type", "unknown")
                )
                control_mode = _safe_control_mode(
                    p_data.get("control_mode", "manipulation")
                )

                primitive = MotionPrimitive(
                    primitive_id=f"prim_{uuid.uuid4().hex[:8]}",
                    action_type=action_type,
                    control_mode=control_mode,
                    start_frame=poses[start_idx].frame_number,
                    end_frame=poses[end_idx].frame_number,
                    duration_seconds=max(end_time - start_time, 0.1),
                    description=p_data.get("description", ""),
                    reasoning=p_data.get("reasoning", ""),
                    object_interactions=p_data.get("object_interactions", []),
                    confidence=float(p_data.get("confidence", 0.5)),
                )
                primitives.append(primitive)

            print(f"[SkillAgent] Segmented {len(primitives)} primitives")
            return primitives

        except Exception as e:
            print(f"[SkillAgent] Segmentation error: {e}")
            return self._fallback_segmentation(poses)

    def add_to_library(self, primitive: MotionPrimitive):
        self.skill_library[primitive.primitive_id] = primitive
        self._save_library()
        print(
            f"[SkillAgent] Library += {primitive.action_type.value} "
            f"({primitive.primitive_id})"
        )

    def compose_skill_sequence(self, task: str) -> SkillSequence:
        library_summary = [
            {
                "id": pid,
                "action": p.action_type.value,
                "mode": p.control_mode.value,
                "description": p.description,
                "duration": p.duration_seconds,
            }
            for pid, p in self.skill_library.items()
        ]

        prompt = SKILL_COMPOSE_PROMPT.format(
            skill_library=json.dumps(library_summary, indent=2),
            task=task,
        )

        try:
            response = self.client.call_text(
                system="You are a robot task planning expert.",
                user=prompt,
                model=self.reasoning_model,
                temperature=0.1,
            )
            data = self.client.extract_json(response)
            skill_ids = data.get("skill_ids", [])
            selected = [
                self.skill_library[sid]
                for sid in skill_ids
                if sid in self.skill_library
            ]
            return SkillSequence(
                task_description=task,
                primitives=selected,
                estimated_duration=sum(p.duration_seconds for p in selected),
                requires_locomotion=bool(data.get("requires_locomotion", False)),
                requires_manipulation=bool(data.get("requires_manipulation", True)),
            )
        except Exception as e:
            print(f"[SkillAgent] Composition error: {e}")
            return SkillSequence(task_description=task, primitives=[])

    def _summarize_poses(self, poses: list[HumanPose]) -> dict:
        loco = sum(1 for p in poses if p.is_moving)
        whole = sum(
            1 for p in poses if p.dominant_mode == ControlMode.WHOLE_BODY
        )
        indices = np.linspace(0, len(poses) - 1, min(5, len(poses)), dtype=int)
        key_poses = [
            {
                "frame": poses[i].frame_number,
                "time": round(poses[i].timestamp, 2),
                "mode": poses[i].dominant_mode.value,
                "is_moving": poses[i].is_moving,
                "has_left_hand": len(poses[i].left_hand) > 0,
                "has_right_hand": len(poses[i].right_hand) > 0,
            }
            for i in indices
        ]
        return {
            "total_frames": len(poses),
            "locomotion_frames": loco,
            "whole_body_frames": whole,
            "locomotion_ratio": round(loco / max(len(poses), 1), 2),
            "key_poses": key_poses,
        }

    def _fallback_segmentation(self, poses: list[HumanPose]) -> list[MotionPrimitive]:
        if not poses:
            return []
        primitives = []
        current_mode = poses[0].dominant_mode
        start_idx = 0
        for i, pose in enumerate(poses[1:], 1):
            if pose.dominant_mode != current_mode or i == len(poses) - 1:
                duration = poses[i].timestamp - poses[start_idx].timestamp
                if duration > 0.5:
                    primitives.append(MotionPrimitive(
                        primitive_id=f"prim_{uuid.uuid4().hex[:8]}",
                        action_type=ActionType.UNKNOWN,
                        control_mode=current_mode,
                        start_frame=poses[start_idx].frame_number,
                        end_frame=poses[i].frame_number,
                        duration_seconds=duration,
                        description=f"Auto-detected {current_mode.value}",
                        reasoning="Rule-based fallback",
                        confidence=0.3,
                    ))
                start_idx = i
                current_mode = pose.dominant_mode
        return primitives

    def _load_library(self):
        path = Path(config.skill.db_path)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for pid, p_data in data.items():
                    self.skill_library[pid] = MotionPrimitive(**p_data)
                print(f"[SkillAgent] Loaded {len(self.skill_library)} skills")
            except Exception as e:
                print(f"[SkillAgent] Library load error: {e}")

    def _save_library(self):
        path = Path(config.skill.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {pid: p.model_dump() for pid, p in self.skill_library.items()}
        path.write_text(json.dumps(data, indent=2, default=str))
