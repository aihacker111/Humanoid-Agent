"""
agents/retargeting_agent.py — LLM-Guided Kinematic Retargeting
OPTIMIZED: Batch calls + keyframe extraction + linear interpolation

Speed comparison:
  Before: 41 poses × 1 call × ~20s = ~820s
  After:  keyframes only (5-8) × 1 batch call × ~25s = ~25-40s
  Speedup: ~20x faster, ~85% fewer tokens
"""
import json
import math
import re
import numpy as np
from typing import Optional

from models import HumanPose, RetargetedPose, ControlMode
from core.openrouter import OpenRouterClient, _extract_json_safe
from config import config


# ── Compact system prompt (fewer tokens) ──────────────────────────────────────
SYSTEM_PROMPT = """You are a kinematic retargeting expert for Unitree H1 robot (27 DOF).
Map human poses to joint angles. Return ONLY valid JSON, no extra text.

Joint limits (rad): hip_pitch/roll/yaw[-0.43,0.43], knee[-0.26,2.05],
ankle[-0.87,0.52], torso[-2.35,2.35], shoulder_pitch[-3.14,3.14],
shoulder_roll/yaw[-1.57,1.57], elbow[-1.57,1.57], wrist_roll[-1.57,1.57]."""


# ── Batch prompt: retarget N poses in one call ─────────────────────────────────
BATCH_PROMPT = """Task: "{task}"
Control mode: {mode}

Retarget these {n} key poses to H1 joint angles.
For each pose, replicate the human's motion INTENT, maintain balance.

Poses:
{poses_json}

Return a JSON array of exactly {n} objects:
[
  {{
    "frame": <frame_number>,
    "joint_angles": {{
      "left_hip_yaw":0.0,"left_hip_roll":0.0,"left_hip_pitch":0.0,
      "left_knee":0.0,"left_ankle":0.0,
      "right_hip_yaw":0.0,"right_hip_roll":0.0,"right_hip_pitch":0.0,
      "right_knee":0.0,"right_ankle":0.0,"torso":0.0,
      "left_shoulder_pitch":0.0,"left_shoulder_roll":0.0,"left_shoulder_yaw":0.0,
      "left_elbow":0.0,"left_wrist_roll":0.0,
      "right_shoulder_pitch":0.0,"right_shoulder_roll":0.0,"right_shoulder_yaw":0.0,
      "right_elbow":0.0,"right_wrist_roll":0.0
    }},
    "balance_score": 0.85,
    "reasoning": "brief"
  }}
]"""


# Neutral standing pose used as fallback and interpolation base
NEUTRAL_POSE: dict[str, float] = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": 0.0,
    "left_knee": 0.0, "left_ankle": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": 0.0,
    "right_knee": 0.0, "right_ankle": 0.0, "torso": 0.0,
    "left_shoulder_pitch": 0.0, "left_shoulder_roll": 0.0, "left_shoulder_yaw": 0.0,
    "left_elbow": 0.0, "left_wrist_roll": 0.0,
    "right_shoulder_pitch": 0.0, "right_shoulder_roll": 0.0, "right_shoulder_yaw": 0.0,
    "right_elbow": 0.0, "right_wrist_roll": 0.0,
}


class RetargetingAgent:
    """
    Fast LLM-guided retargeting:
      1. Extract keyframes (significant pose changes only)
      2. Retarget keyframes in ONE batched API call
      3. Interpolate remaining frames — no extra API calls
    """

    def __init__(self):
        self.client = OpenRouterClient()
        self.model = config.openrouter.reasoning_model
        self.joint_limits = config.humanoid.joint_limits

    # ── Public API ─────────────────────────────────────────────────────────────

    def retarget_sequence(
        self,
        poses: list[HumanPose],
        task_context: str = "",
        max_keyframes: int = 8,
        batch_size: int = 8,
    ) -> list[RetargetedPose]:
        """
        Main entry point — fast batch retargeting.

        Args:
            poses:          All extracted human poses
            task_context:   Task description
            max_keyframes:  Max poses to send to LLM (default 8)
            batch_size:     Poses per API call (default 8 = 1 call for ≤8 keyframes)

        Returns:
            Full retargeted sequence (all frames, interpolated)
        """
        if not poses:
            return []

        print(f"  [Retarget] {len(poses)} total poses → extracting keyframes...")

        # Step 1: Select keyframes (reduces API calls drastically)
        keyframe_indices = self._select_keyframes(poses, max_keyframes)
        keyframe_poses = [poses[i] for i in keyframe_indices]
        print(f"  [Retarget] {len(keyframe_poses)} keyframes selected "
              f"({len(keyframe_poses)} API calls → 1 batch call)")

        # Step 2: Batch retarget all keyframes in one API call
        keyframe_results = self._retarget_batch(
            keyframe_poses, task_context, batch_size
        )

        # Step 3: Interpolate between keyframes for all other frames
        full_sequence = self._interpolate_full(
            poses, keyframe_indices, keyframe_results
        )

        # Step 4: Smooth
        return self._smooth(full_sequence)

    # ── Keyframe selection ─────────────────────────────────────────────────────

    def _select_keyframes(
        self,
        poses: list[HumanPose],
        max_keyframes: int,
    ) -> list[int]:
        """
        Select the most representative frames by pose change magnitude.
        Always includes first and last frame.
        """
        if len(poses) <= max_keyframes:
            return list(range(len(poses)))

        # Compute pose change score for each frame
        scores = [0.0]
        for i in range(1, len(poses)):
            score = self._pose_change_score(poses[i - 1], poses[i])
            scores.append(score)

        # Always include first and last
        selected = {0, len(poses) - 1}

        # Add frames with highest change scores
        indexed_scores = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        for idx, _ in indexed_scores:
            if len(selected) >= max_keyframes:
                break
            selected.add(idx)

        return sorted(selected)

    def _pose_change_score(self, p1: HumanPose, p2: HumanPose) -> float:
        """Measure how much pose changed between two frames."""
        total = 0.0
        count = 0
        key_joints = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
        ]
        for joint in key_joints:
            kp1 = p1.body.get(joint)
            kp2 = p2.body.get(joint)
            if kp1 and kp2:
                dx = kp1.x - kp2.x
                dy = kp1.y - kp2.y
                dz = kp1.z - kp2.z
                total += math.sqrt(dx*dx + dy*dy + dz*dz)
                count += 1
        return total / max(count, 1)

    # ── Batch API call ─────────────────────────────────────────────────────────

    def _retarget_batch(
        self,
        poses: list[HumanPose],
        task_context: str,
        batch_size: int,
    ) -> list[RetargetedPose]:
        """Send all keyframes in batches of `batch_size` per API call."""
        results = []
        total_batches = math.ceil(len(poses) / batch_size)

        for batch_idx in range(0, len(poses), batch_size):
            batch = poses[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            print(f"  [Retarget] Batch {batch_num}/{total_batches} "
                  f"({len(batch)} poses)...")

            batch_results = self._call_batch_api(batch, task_context)
            results.extend(batch_results)

        return results

    def _call_batch_api(
        self,
        poses: list[HumanPose],
        task_context: str,
    ) -> list[RetargetedPose]:
        """Single API call retargeting multiple poses."""

        # Compact pose representation to save tokens
        poses_data = [
            {
                "frame": p.frame_number,
                "key_joints": self._format_pose_compact(p),
                "moving": p.is_moving,
                "has_hands": bool(p.left_hand or p.right_hand),
            }
            for p in poses
        ]

        # Detect dominant mode across batch
        modes = [p.dominant_mode.value for p in poses]
        dominant_mode = max(set(modes), key=modes.count)

        prompt = BATCH_PROMPT.format(
            task=task_context or "general motion",
            mode=dominant_mode,
            n=len(poses),
            poses_json=json.dumps(poses_data, indent=1),
        )

        try:
            response = self.client.call_text(
                system=SYSTEM_PROMPT,
                user=prompt,
                model=self.model,
                temperature=0.05,
                max_tokens=4096,
            )
            text = self.client.extract_text(response)

            # Extract JSON array robustly
            batch_data = self._extract_json_array(text)

            results = []
            for i, pose in enumerate(poses):
                if i < len(batch_data):
                    data = batch_data[i]
                    angles = self._clamp(data.get("joint_angles", {}))
                    if not angles:
                        angles = NEUTRAL_POSE.copy()
                    results.append(RetargetedPose(
                        frame_number=pose.frame_number,
                        source_human_pose=pose,
                        joint_angles=angles,
                        balance_score=float(data.get("balance_score", 0.8)),
                        reachability_score=0.85,
                        retargeting_error=self._compute_error(pose, angles),
                        retargeting_reasoning=data.get("reasoning", "batch"),
                    ))
                else:
                    results.append(self._fallback(pose))

            return results

        except Exception as e:
            print(f"  [Retarget] Batch API error: {e}")
            return [self._fallback(p) for p in poses]

    # ── Interpolation ──────────────────────────────────────────────────────────

    def _interpolate_full(
        self,
        all_poses: list[HumanPose],
        keyframe_indices: list[int],
        keyframe_results: list[RetargetedPose],
    ) -> list[RetargetedPose]:
        """
        Fill in all frames between keyframes using linear interpolation.
        No API calls — pure math.
        """
        if not keyframe_results:
            return [self._fallback(p) for p in all_poses]

        # Build lookup: frame_number → RetargetedPose
        kf_map: dict[int, RetargetedPose] = {
            r.frame_number: r for r in keyframe_results
        }
        kf_frames = sorted(kf_map.keys())

        full = []
        for i, pose in enumerate(all_poses):
            fn = pose.frame_number

            # Exact keyframe
            if fn in kf_map:
                full.append(kf_map[fn])
                continue

            # Find surrounding keyframes
            before_fn = max((f for f in kf_frames if f <= fn), default=kf_frames[0])
            after_fn  = min((f for f in kf_frames if f >= fn), default=kf_frames[-1])

            if before_fn == after_fn:
                # Copy nearest keyframe
                r = kf_map[before_fn]
                full.append(RetargetedPose(
                    frame_number=fn,
                    source_human_pose=pose,
                    joint_angles=r.joint_angles.copy(),
                    balance_score=r.balance_score,
                    reachability_score=r.reachability_score,
                    retargeting_error=r.retargeting_error,
                    retargeting_reasoning="interpolated (copy)",
                ))
                continue

            # Linear interpolation
            t = (fn - before_fn) / (after_fn - before_fn)
            r_before = kf_map[before_fn]
            r_after  = kf_map[after_fn]

            interp_angles = {
                joint: r_before.joint_angles.get(joint, 0.0) * (1 - t)
                       + r_after.joint_angles.get(joint, 0.0) * t
                for joint in NEUTRAL_POSE
            }
            interp_angles = self._clamp(interp_angles)

            full.append(RetargetedPose(
                frame_number=fn,
                source_human_pose=pose,
                joint_angles=interp_angles,
                balance_score=(r_before.balance_score * (1-t) + r_after.balance_score * t),
                reachability_score=0.85,
                retargeting_error=self._compute_error(pose, interp_angles),
                retargeting_reasoning=f"interpolated t={t:.2f}",
            ))

        return full

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _format_pose_compact(self, pose: HumanPose) -> dict:
        """Compact pose format — only key joints, 2 decimal places."""
        key_joints = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
            "nose",
        ]
        result = {}
        for name in key_joints:
            kp = pose.body.get(name)
            if kp and kp.visibility > 0.5:
                result[name] = {
                    "x": round(kp.x, 2),
                    "y": round(kp.y, 2),
                    "z": round(kp.z, 2),
                }
        # Add hand presence summary (not full 21 points — saves tokens)
        if pose.left_hand:
            wrist = pose.left_hand.get("wrist")
            if wrist:
                result["left_hand_wrist"] = {
                    "x": round(wrist.x, 2), "y": round(wrist.y, 2)
                }
        if pose.right_hand:
            wrist = pose.right_hand.get("wrist")
            if wrist:
                result["right_hand_wrist"] = {
                    "x": round(wrist.x, 2), "y": round(wrist.y, 2)
                }
        return result

    def _extract_json_array(self, text: str) -> list:
        """Extract JSON array from LLM response robustly."""
        text = text.strip()

        # Remove markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Try direct parse
        try:
            result = json.loads(text)
            return result if isinstance(result, list) else [result]
        except json.JSONDecodeError:
            pass

        # Find array by bracket matching
        start = text.find("[")
        if start == -1:
            raise ValueError(f"No JSON array found: {text[:200]}")

        depth = 0
        end = -1
        for i, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == -1:
            # Try to close open brackets
            open_sq = text.count("[") - text.count("]")
            open_cu = text.count("{") - text.count("}")
            fixed = text[start:] + "}" * open_cu + "]" * open_sq
            try:
                return json.loads(fixed)
            except Exception:
                raise ValueError(f"Malformed JSON array: {text[start:start+300]}")

        return json.loads(text[start:end])

    def _clamp(self, angles: dict) -> dict:
        clamped = {}
        for joint, angle in angles.items():
            limit_key = next(
                (k for k in self.joint_limits if k in joint), None
            )
            if limit_key:
                lo, hi = self.joint_limits[limit_key]
                clamped[joint] = float(np.clip(angle, lo, hi))
            else:
                clamped[joint] = float(angle)
        return clamped

    def _compute_error(self, pose: HumanPose, angles: dict) -> float:
        le = pose.body.get("left_elbow")
        re = pose.body.get("right_elbow")
        err = 0.0
        if le:
            err += abs(abs(le.y - 0.5) * math.pi - abs(angles.get("left_elbow", 0.0)))
        if re:
            err += abs(abs(re.y - 0.5) * math.pi - abs(angles.get("right_elbow", 0.0)))
        return round(float(err / 2), 4)

    def _smooth(
        self, poses: list[RetargetedPose], window: int = 3
    ) -> list[RetargetedPose]:
        if len(poses) < window:
            return poses
        smoothed = []
        for i, pose in enumerate(poses):
            start = max(0, i - window // 2)
            end = min(len(poses), i + window // 2 + 1)
            wp = poses[start:end]
            angles = {
                joint: float(np.mean([p.joint_angles.get(joint, 0.0) for p in wp]))
                for joint in NEUTRAL_POSE
            }
            angles = self._clamp(angles)
            smoothed.append(RetargetedPose(
                frame_number=pose.frame_number,
                source_human_pose=pose.source_human_pose,
                joint_angles=angles,
                balance_score=pose.balance_score,
                reachability_score=pose.reachability_score,
                retargeting_error=pose.retargeting_error,
                retargeting_reasoning=pose.retargeting_reasoning,
            ))
        return smoothed

    def _fallback(self, human_pose: HumanPose) -> RetargetedPose:
        return RetargetedPose(
            frame_number=human_pose.frame_number,
            source_human_pose=human_pose,
            joint_angles=NEUTRAL_POSE.copy(),
            retargeting_error=1.0,
            balance_score=0.5,
            reachability_score=0.5,
            retargeting_reasoning="fallback neutral pose",
        )

    # Keep single-pose method for backward compatibility
    def retarget(
        self,
        human_pose: HumanPose,
        task_context: str = "",
    ) -> RetargetedPose:
        results = self._retarget_batch([human_pose], task_context, batch_size=1)
        return results[0] if results else self._fallback(human_pose)
