"""
agents/retargeting_agent.py — LLM-Guided Retargeting with Embedded IK Formulas
v2: Mathematical formulas embedded in system prompt → LLM computes joint angles
"""
import json
import re
import numpy as np
from models import HumanPose, RetargetedPose, ControlMode
from core.openrouter import OpenRouterClient, _extract_json_safe
from config import config


# ── System prompt with full kinematic formulas ─────────────────────────────────
SYSTEM_PROMPT = """You are a kinematic retargeting expert for the Unitree H1 humanoid robot.

## YOUR TASK
Convert MediaPipe body landmarks (image space) to H1 joint angles (radians).

## INPUT FORMAT
MediaPipe landmarks: normalized image coordinates
  x ∈ [0,1]: horizontal (0=left, 1=right)
  y ∈ [0,1]: vertical   (0=top,  1=bottom) ← y increases DOWNWARD
  z ∈ real:  depth estimate (negative = closer to camera)
  visibility ∈ [0,1]: confidence

## MATHEMATICAL FORMULAS — APPLY THESE EXACTLY

### Vector computation
Given two landmarks A and B:
  vector AB = [B.x - A.x, B.y - A.y, B.z - A.z]
  |AB| = sqrt((B.x-A.x)² + (B.y-A.y)² + (B.z-A.z)²)

### Angle between two vectors (bend angle)
  angle = arccos(dot(v1,v2) / (|v1| × |v2|))
  dot(v1,v2) = v1.x×v2.x + v1.y×v2.y + v1.z×v2.z

### ELBOW ANGLE (left and right)
  upper_arm = vector(shoulder → elbow)
  forearm   = vector(elbow → wrist)
  elbow_angle = π - angle_between(upper_arm, forearm)
  Clamp to [0.0, 1.57]

### SHOULDER PITCH (forward/backward elevation)
  arm_vec = normalize(vector(shoulder → elbow))
  shoulder_pitch = -arctan2(arm_vec.z, arm_vec.y)
  Scale by 0.5. Left: use +result, Right: use -result
  Clamp to [-3.14, 3.14]

### SHOULDER ROLL (lateral abduction)
  arm_vec = normalize(vector(shoulder → elbow))
  roll = arctan2(arm_vec.x, -arm_vec.y)
  Left: base=+0.15 + 0.4×roll, Right: base=-0.15 - 0.4×roll
  Clamp to [-1.57, 1.57]

### SHOULDER YAW (arm rotation)
  upper = normalize(vector(shoulder → elbow))
  fore  = normalize(vector(elbow → wrist))
  cross = upper × fore  (cross product)
  yaw = arctan2(cross.x, cross.z) × 0.3
  Clamp to [-1.57, 1.57]

### HIP PITCH (forward/backward leg swing)
  thigh = normalize(vector(hip → knee))
  hip_pitch = -arctan2(thigh.z, thigh.y) × 0.5
  Clamp to [-0.43, 0.43]

### HIP ROLL (lateral leg swing)
  thigh = normalize(vector(hip → knee))
  roll = arctan2(thigh.x, -thigh.y)
  Left: +roll×0.3, Right: -roll×0.3
  Clamp to [-0.43, 0.43]

### KNEE ANGLE
  thigh = vector(hip → knee)
  shin  = vector(knee → ankle)
  knee = |π - angle_between(thigh, shin)|
  Clamp to [0.0, 2.05]

### ANKLE ANGLE
  If foot_index available:
    shin = vector(knee → ankle)
    foot = vector(ankle → foot_index)
    ankle = angle_between(shin, foot) - π/2
  Else: use -0.34 (neutral standing)
  Clamp to [-0.87, 0.52]

### TORSO YAW
  shoulder_vec = vector(left_shoulder → right_shoulder)
  hip_vec      = vector(left_hip → right_hip)
  s_angle = arctan2(shoulder_vec.z, shoulder_vec.x)
  h_angle = arctan2(hip_vec.z, hip_vec.x)
  torso = (s_angle - h_angle) × 0.5
  Clamp to [-2.35, 2.35]

## JOINT LIMITS SUMMARY
hip_pitch/roll/yaw: [-0.43, 0.43]
knee: [0.0, 2.05]  ankle: [-0.87, 0.52]  torso: [-2.35, 2.35]
shoulder_pitch: [-3.14, 3.14]  shoulder_roll: [-1.57, 1.57]
shoulder_yaw: [-1.57, 1.57]  elbow: [0.0, 1.57]  wrist_roll: [-1.57, 1.57]

## RULES
1. Apply ALL formulas above precisely
2. If landmark visibility < 0.3, use neutral value (0.0 for most, see defaults)
3. Clamp ALL angles to their limits
4. Return ONLY valid JSON, no extra text"""


RETARGET_PROMPT = """Task context: {task}
Control mode: {mode}

Compute H1 joint angles for these {n} poses using the formulas in your instructions.
For each pose, show your reasoning then give the final angles.

Poses:
{poses_json}

Return ONLY a JSON array of {n} objects:
[
  {{
    "frame": <frame_number>,
    "reasoning": "applied formulas: elbow=π-angle(upper_arm,forearm)=...",
    "joint_angles": {{
      "left_hip_yaw":0.0, "left_hip_roll":0.0, "left_hip_pitch":0.0,
      "left_knee":0.0, "left_ankle":0.0,
      "right_hip_yaw":0.0, "right_hip_roll":0.0, "right_hip_pitch":0.0,
      "right_knee":0.0, "right_ankle":0.0, "torso":0.0,
      "left_shoulder_pitch":0.0, "left_shoulder_roll":0.15,
      "left_shoulder_yaw":0.0, "left_elbow":0.3, "left_wrist_roll":0.0,
      "right_shoulder_pitch":0.0, "right_shoulder_roll":-0.15,
      "right_shoulder_yaw":0.0, "right_elbow":0.3, "right_wrist_roll":0.0
    }},
    "balance_score": 0.85,
    "reachability_score": 0.9
  }}
]"""


NEUTRAL_POSE = {
    "left_hip_yaw":0.0,"left_hip_roll":0.0,"left_hip_pitch":0.0,
    "left_knee":0.0,"left_ankle":0.0,
    "right_hip_yaw":0.0,"right_hip_roll":0.0,"right_hip_pitch":0.0,
    "right_knee":0.0,"right_ankle":0.0,"torso":0.0,
    "left_shoulder_pitch":0.0,"left_shoulder_roll":0.15,
    "left_shoulder_yaw":0.0,"left_elbow":0.3,"left_wrist_roll":0.0,
    "right_shoulder_pitch":0.0,"right_shoulder_roll":-0.15,
    "right_shoulder_yaw":0.0,"right_elbow":0.3,"right_wrist_roll":0.0,
}

JOINT_LIMITS = {
    "hip_pitch":(-0.43,0.43),"hip_roll":(-0.43,0.43),"hip_yaw":(-0.43,0.43),
    "knee":(0.0,2.05),"ankle":(-0.87,0.52),"torso":(-2.35,2.35),
    "shoulder_pitch":(-3.14,3.14),"shoulder_roll":(-1.57,1.57),
    "shoulder_yaw":(-1.57,1.57),"elbow":(0.0,1.57),"wrist_roll":(-1.57,1.57),
}


class RetargetingAgent:
    """
    LLM retargeting with embedded kinematic formulas.
    The LLM receives the full IK formulas in its system prompt
    and computes joint angles mathematically.
    """

    def __init__(self):
        self.client = OpenRouterClient()
        self.model = config.openrouter.reasoning_model
        self.joint_limits = JOINT_LIMITS

    def retarget_sequence(
        self,
        poses: list[HumanPose],
        task_context: str = "",
        max_keyframes: int = 8,
        batch_size: int = 8,
    ) -> list[RetargetedPose]:
        if not poses:
            return []

        # Select keyframes
        keyframe_indices = self._select_keyframes(poses, max_keyframes)
        keyframe_poses = [poses[i] for i in keyframe_indices]
        print(f"  [Retarget] {len(poses)} poses → {len(keyframe_poses)} keyframes → 1 batch call")

        # Batch retarget keyframes
        keyframe_results = self._batch_retarget(keyframe_poses, task_context)

        # Interpolate all frames
        full = self._interpolate(poses, keyframe_indices, keyframe_results)
        return self._smooth(full)

    def _select_keyframes(self, poses: list[HumanPose], n: int) -> list[int]:
        if len(poses) <= n:
            return list(range(len(poses)))
        scores = [0.0]
        for i in range(1, len(poses)):
            s = self._change_score(poses[i-1], poses[i])
            scores.append(s)
        selected = {0, len(poses)-1}
        for idx, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True):
            if len(selected) >= n: break
            selected.add(idx)
        return sorted(selected)

    def _change_score(self, p1: HumanPose, p2: HumanPose) -> float:
        total = 0.0; count = 0
        for j in ["left_elbow","right_elbow","left_wrist","right_wrist",
                  "left_knee","right_knee"]:
            k1,k2 = p1.body.get(j), p2.body.get(j)
            if k1 and k2:
                total += abs(k1.x-k2.x) + abs(k1.y-k2.y)
                count += 1
        return total / max(count, 1)

    def _batch_retarget(
        self,
        poses: list[HumanPose],
        task_context: str,
    ) -> list[RetargetedPose]:
        # Format compact pose data for LLM
        poses_data = []
        for p in poses:
            kps = {}
            relevant = [
                "left_shoulder","right_shoulder","left_elbow","right_elbow",
                "left_wrist","right_wrist","left_hip","right_hip",
                "left_knee","right_knee","left_ankle","right_ankle",
                "left_foot_index","right_foot_index","nose",
            ]
            for name in relevant:
                kp = p.body.get(name)
                if kp and kp.visibility > 0.25:
                    kps[name] = {
                        "x": round(kp.x, 3),
                        "y": round(kp.y, 3),
                        "z": round(kp.z, 3),
                        "v": round(kp.visibility, 2),
                    }
            poses_data.append({
                "frame": p.frame_number,
                "landmarks": kps,
                "is_moving": p.is_moving,
            })

        modes = [p.dominant_mode.value for p in poses]
        mode = max(set(modes), key=modes.count)

        prompt = RETARGET_PROMPT.format(
            task=task_context or "general motion",
            mode=mode,
            n=len(poses),
            poses_json=json.dumps(poses_data, indent=1),
        )

        try:
            response = self.client.call_text(
                system=SYSTEM_PROMPT,
                user=prompt,
                model=self.model,
                temperature=0.05,
                max_tokens=6000,  # More tokens for formula reasoning
            )
            text = self.client.extract_text(response)
            batch_data = self._extract_array(text)

            results = []
            for i, pose in enumerate(poses):
                if i < len(batch_data):
                    d = batch_data[i]
                    if not isinstance(d, dict):
                        results.append(self._fallback(pose))
                        continue
                    angles = self._clamp(d.get("joint_angles", {}))
                    if not angles:
                        angles = NEUTRAL_POSE.copy()
                    results.append(RetargetedPose(
                        frame_number=pose.frame_number,
                        source_human_pose=pose,
                        joint_angles=angles,
                        balance_score=float(d.get("balance_score", 0.8)),
                        reachability_score=float(d.get("reachability_score", 0.85)),
                        retargeting_error=0.0,
                        retargeting_reasoning=d.get("reasoning", ""),
                    ))
                else:
                    results.append(self._fallback(pose))
            return results

        except Exception as e:
            print(f"  [Retarget] Batch error: {e}")
            return [self._fallback(p) for p in poses]

    def _extract_array(self, text: str) -> list:
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()
        try:
            r = json.loads(text)
            return r if isinstance(r, list) else [r]
        except Exception:
            pass
        start = text.find("[")
        if start == -1:
            raise ValueError(f"No JSON array: {text[:200]}")
        depth = 0; end = -1
        for i, ch in enumerate(text[start:], start):
            if ch == "[": depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0: end = i+1; break
        if end == -1:
            oc = text.count("[") - text.count("]")
            ob = text.count("{") - text.count("}")
            try: return json.loads(text[start:] + "}"*ob + "]"*oc)
            except: raise ValueError("Malformed JSON array")
        return json.loads(text[start:end])

    def _interpolate(
        self,
        all_poses: list[HumanPose],
        kf_indices: list[int],
        kf_results: list[RetargetedPose],
    ) -> list[RetargetedPose]:
        if not kf_results:
            return [self._fallback(p) for p in all_poses]
        kf_map = {r.frame_number: r for r in kf_results}
        kf_frames = sorted(kf_map.keys())
        full = []
        for pose in all_poses:
            fn = pose.frame_number
            if fn in kf_map:
                full.append(kf_map[fn])
                continue
            before = max((f for f in kf_frames if f <= fn), default=kf_frames[0])
            after  = min((f for f in kf_frames if f >= fn), default=kf_frames[-1])
            if before == after:
                r = kf_map[before]
                full.append(RetargetedPose(
                    frame_number=fn, source_human_pose=pose,
                    joint_angles=r.joint_angles.copy(),
                    balance_score=r.balance_score,
                    reachability_score=r.reachability_score,
                    retargeting_reasoning="interpolated(copy)",
                ))
            else:
                t = (fn - before) / (after - before)
                ra, rb = kf_map[before], kf_map[after]
                angles = {
                    j: ra.joint_angles.get(j, 0.0)*(1-t) + rb.joint_angles.get(j, 0.0)*t
                    for j in NEUTRAL_POSE
                }
                angles = self._clamp(angles)
                full.append(RetargetedPose(
                    frame_number=fn, source_human_pose=pose,
                    joint_angles=angles,
                    balance_score=ra.balance_score*(1-t)+rb.balance_score*t,
                    reachability_score=0.85,
                    retargeting_reasoning=f"interpolated t={t:.2f}",
                ))
        return full

    def _smooth(self, poses: list[RetargetedPose], w: int = 3) -> list[RetargetedPose]:
        if len(poses) < w:
            return poses
        smoothed = []
        for i, pose in enumerate(poses):
            wp = poses[max(0, i-w//2):min(len(poses), i+w//2+1)]
            angles = {
                j: float(np.mean([p.joint_angles.get(j, 0.0) for p in wp]))
                for j in NEUTRAL_POSE
            }
            smoothed.append(RetargetedPose(
                frame_number=pose.frame_number,
                source_human_pose=pose.source_human_pose,
                joint_angles=self._clamp(angles),
                balance_score=pose.balance_score,
                reachability_score=pose.reachability_score,
                retargeting_reasoning=pose.retargeting_reasoning,
            ))
        return smoothed

    def _clamp(self, angles: dict) -> dict:
        out = {}
        for joint, val in angles.items():
            key = next((k for k in self.joint_limits if k in joint), None)
            if key:
                lo, hi = self.joint_limits[key]
                out[joint] = float(np.clip(val, lo, hi))
            else:
                out[joint] = float(val)
        return out

    def _fallback(self, pose: HumanPose) -> RetargetedPose:
        return RetargetedPose(
            frame_number=pose.frame_number,
            source_human_pose=pose,
            joint_angles=NEUTRAL_POSE.copy(),
            balance_score=0.5, reachability_score=0.5,
            retargeting_reasoning="fallback neutral",
        )

    # Backward compat
    def retarget(self, pose: HumanPose, task_context: str = "") -> RetargetedPose:
        return self._batch_retarget([pose], task_context)[0]
