"""
core/trajectory_recorder.py — Save and replay retargeted joint trajectories

Saves:
  - Full joint angle trajectory per session (JSON + NPY)
  - Per-frame retargeting reasoning (Markdown log)
  - Metadata for paper reproducibility
"""
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

from models import RetargetedPose, RetargetedSequence, HumanPose
from config import config


class TrajectoryRecorder:
    """
    Records retargeted trajectories to disk.
    Enables replay, analysis, and paper reproducibility.
    """

    def __init__(self, session_id: str = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(config.output_dir)
        self.traj_dir = self.base_dir / "trajectories"
        self.reasoning_dir = self.base_dir / "reasoning"
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self.reasoning_dir.mkdir(parents=True, exist_ok=True)

        # In-memory buffers
        self._frames: list[RetargetedPose] = []
        self._reasoning_log: list[dict] = []
        self._task: str = ""
        self._source_video: str = ""
        self._start_time: float = time.time()

    def start_session(self, task: str, source_video: str = ""):
        """Start recording a new session"""
        self._task = task
        self._source_video = source_video
        self._frames = []
        self._reasoning_log = []
        self._start_time = time.time()
        print(f"[Recorder] Session started: {self.session_id}")

    def record_frame(self, pose: RetargetedPose):
        """Record a single retargeted frame"""
        self._frames.append(pose)

        # Log reasoning if present
        if pose.retargeting_reasoning:
            self._reasoning_log.append({
                "frame": pose.frame_number,
                "timestamp": pose.source_human_pose.timestamp if pose.source_human_pose else 0.0,
                "mode": pose.source_human_pose.dominant_mode.value if pose.source_human_pose else "unknown",
                "reasoning": pose.retargeting_reasoning,
                "balance_score": pose.balance_score,
                "reachability_score": pose.reachability_score,
                "retargeting_error": pose.retargeting_error,
                "constraint_violations": pose.constraint_violations,
                "joint_angles": pose.joint_angles,
            })

    def record_sequence(self, sequence: RetargetedSequence):
        """Record an entire sequence at once"""
        for frame in sequence.frames:
            self.record_frame(frame)

    def save(self) -> dict:
        """
        Save all recorded data to disk.
        Returns paths to saved files.
        """
        if not self._frames:
            print("[Recorder] Nothing to save.")
            return {}

        saved_paths = {}

        # ── 1. Joint trajectory as JSON ───────────────────────────────────────
        traj_json_path = self.traj_dir / f"{self.session_id}_trajectory.json"
        trajectory_data = {
            "session_id": self.session_id,
            "task": self._task,
            "source_video": self._source_video,
            "model_used": config.openrouter.reasoning_model,
            "total_frames": len(self._frames),
            "duration_seconds": time.time() - self._start_time,
            "joint_names": list(self._frames[0].joint_angles.keys()) if self._frames else [],
            "frames": [
                {
                    "frame_number": f.frame_number,
                    "timestamp": f.source_human_pose.timestamp if f.source_human_pose else 0.0,
                    "joint_angles": f.joint_angles,
                    "balance_score": f.balance_score,
                    "reachability_score": f.reachability_score,
                    "retargeting_error": f.retargeting_error,
                }
                for f in self._frames
            ],
        }
        traj_json_path.write_text(json.dumps(trajectory_data, indent=2))
        saved_paths["trajectory_json"] = str(traj_json_path)

        # ── 2. Joint trajectory as NPY (fast loading for replay) ──────────────
        traj_npy_path = self.traj_dir / f"{self.session_id}_trajectory.npy"
        if self._frames and self._frames[0].joint_angles:
            joint_names = list(self._frames[0].joint_angles.keys())
            matrix = np.array([
                [f.joint_angles.get(j, 0.0) for j in joint_names]
                for f in self._frames
            ])
            np.save(str(traj_npy_path), matrix)
            saved_paths["trajectory_npy"] = str(traj_npy_path)

            # Save joint name mapping
            meta_path = self.traj_dir / f"{self.session_id}_joint_names.json"
            meta_path.write_text(json.dumps(joint_names))

        # ── 3. Per-frame reasoning log (Markdown for human reading) ───────────
        if self._reasoning_log:
            reasoning_md_path = self.reasoning_dir / f"{self.session_id}_reasoning.md"
            reasoning_md_path.write_text(self._build_reasoning_markdown())
            saved_paths["reasoning_md"] = str(reasoning_md_path)

            reasoning_json_path = self.reasoning_dir / f"{self.session_id}_reasoning.json"
            reasoning_json_path.write_text(json.dumps(self._reasoning_log, indent=2))
            saved_paths["reasoning_json"] = str(reasoning_json_path)

        # ── 4. Quality metrics summary ────────────────────────────────────────
        metrics_path = self.traj_dir / f"{self.session_id}_metrics.json"
        metrics = self._compute_quality_metrics()
        metrics_path.write_text(json.dumps(metrics, indent=2))
        saved_paths["metrics"] = str(metrics_path)

        print(f"[Recorder] Saved {len(self._frames)} frames to {self.traj_dir}")
        for name, path in saved_paths.items():
            print(f"  [{name}] {path}")

        return saved_paths

    def _build_reasoning_markdown(self) -> str:
        """Build human-readable reasoning log for analysis"""
        lines = [
            f"# Retargeting Reasoning Log",
            f"**Session:** {self.session_id}",
            f"**Task:** {self._task}",
            f"**Model:** {config.openrouter.reasoning_model}",
            f"**Frames:** {len(self._reasoning_log)}",
            "",
            "---",
            "",
        ]
        for entry in self._reasoning_log:
            lines += [
                f"## Frame {entry['frame']} — t={entry['timestamp']:.2f}s",
                f"**Mode:** `{entry['mode']}` | "
                f"**Balance:** {entry['balance_score']:.2f} | "
                f"**Reachability:** {entry['reachability_score']:.2f} | "
                f"**Error:** {entry['retargeting_error']:.3f}",
                "",
                "**LLM Reasoning:**",
                f"> {entry['reasoning']}",
                "",
            ]
            if entry["constraint_violations"]:
                lines += [
                    "**Constraint violations:**",
                    *[f"- {v}" for v in entry["constraint_violations"]],
                    "",
                ]

            # Key joint angles
            angles = entry["joint_angles"]
            key_joints = ["left_knee", "right_knee", "left_elbow", "right_elbow", "torso"]
            angle_str = " | ".join(
                f"`{j}`: {angles.get(j, 0.0):.3f}" for j in key_joints if j in angles
            )
            lines += [f"**Key joints:** {angle_str}", "", "---", ""]

        return "\n".join(lines)

    def _compute_quality_metrics(self) -> dict:
        if not self._frames:
            return {}

        balances = [f.balance_score for f in self._frames]
        reaches = [f.reachability_score for f in self._frames]
        errors = [f.retargeting_error for f in self._frames]
        violations = sum(len(f.constraint_violations) for f in self._frames)

        return {
            "session_id": self.session_id,
            "task": self._task,
            "total_frames": len(self._frames),
            "avg_balance_score": round(float(np.mean(balances)), 4),
            "avg_reachability_score": round(float(np.mean(reaches)), 4),
            "avg_retargeting_error": round(float(np.mean(errors)), 4),
            "total_constraint_violations": violations,
            "frames_with_violations": sum(1 for f in self._frames if f.constraint_violations),
            "model_used": config.openrouter.reasoning_model,
        }


class TrajectoryPlayer:
    """
    Replay saved trajectories in simulation.
    Useful for: debugging, generating paper figures, side-by-side comparison.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.traj_dir = Path(config.output_dir) / "trajectories"

    def load_trajectory(self) -> tuple[np.ndarray, list[str]]:
        """
        Load trajectory matrix and joint names.
        Returns: (matrix [T x J], joint_names [J])
        """
        npy_path = self.traj_dir / f"{self.session_id}_trajectory.npy"
        names_path = self.traj_dir / f"{self.session_id}_joint_names.json"

        if not npy_path.exists():
            raise FileNotFoundError(f"Trajectory not found: {npy_path}")

        matrix = np.load(str(npy_path))
        joint_names = json.loads(names_path.read_text()) if names_path.exists() else []

        print(f"[Player] Loaded trajectory: {matrix.shape[0]} frames, {matrix.shape[1]} joints")
        return matrix, joint_names

    def replay_in_simulation(self, env, show_viewer: bool = True) -> list:
        """Replay saved trajectory in Genesis simulator"""
        matrix, joint_names = self.load_trajectory()

        trajectory = [
            {joint_names[j]: float(matrix[t, j]) for j in range(len(joint_names))}
            for t in range(matrix.shape[0])
        ]

        print(f"[Player] Replaying {len(trajectory)} frames...")
        frames = env.execute_sequence(trajectory, render_frames=True)
        return frames

    def list_sessions(self) -> list[str]:
        """List all saved sessions"""
        sessions = []
        for f in self.traj_dir.glob("*_trajectory.json"):
            sessions.append(f.stem.replace("_trajectory", ""))
        return sorted(sessions)
