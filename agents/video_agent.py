"""
agents/video_agent.py — VLM Scene Understanding Agent
Fixed: robust JSON extraction, handles extra text from LLM
"""
import numpy as np
from models import SceneContext
from core.openrouter import OpenRouterClient, _extract_json_safe
from config import config


SCENE_PROMPT = """
Analyze this video frame for robot learning purposes.

Return ONLY a valid JSON object, no markdown, no extra text:
{
  "description": "1-2 sentence scene overview",
  "detected_objects": ["list of objects"],
  "human_activity": "what the person is doing",
  "locomotion_observed": false,
  "manipulation_observed": true,
  "environment_type": "kitchen|workshop|outdoor|lab|gym|other",
  "task_goal": "inferred goal"
}
"""

JUDGE_PROMPT = """
Evaluate this humanoid robot execution frame.
Task: "{task}"

Return ONLY valid JSON:
{{
  "motion_naturalness": 0.0,
  "task_progress": 0.0,
  "balance_quality": 0.0,
  "manipulation_quality": 0.0,
  "observations": "brief description"
}}

Score each metric 0.0 to 1.0.
"""


class VideoUnderstandingAgent:
    def __init__(self):
        self.client = OpenRouterClient()
        self.model = config.openrouter.vision_model

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
    ) -> SceneContext:
        try:
            response = self.client.call_vision(
                prompt=SCENE_PROMPT,
                frame=frame,
                model=self.model,
            )
            data = self.client.extract_json(response)
            return SceneContext(
                frame_number=frame_number,
                description=data.get("description", ""),
                detected_objects=data.get("detected_objects", []),
                human_activity=data.get("human_activity", ""),
                locomotion_observed=bool(data.get("locomotion_observed", False)),
                manipulation_observed=bool(data.get("manipulation_observed", False)),
                environment_type=data.get("environment_type", "unknown"),
                task_goal=data.get("task_goal"),
            )
        except Exception as e:
            print(f"[VideoAgent] Error: {e}")
            return SceneContext(
                frame_number=frame_number,
                description="Analysis failed",
            )

    def judge_execution(
        self,
        frame: np.ndarray,
        task: str,
        frame_number: int,
    ) -> dict:
        try:
            response = self.client.call_vision(
                prompt=JUDGE_PROMPT.format(task=task),
                frame=frame,
                model=self.model,
            )
            return self.client.extract_json(response)
        except Exception:
            return {
                "motion_naturalness": 0.0,
                "task_progress": 0.0,
                "balance_quality": 0.0,
                "manipulation_quality": 0.0,
                "observations": "Evaluation failed",
            }
