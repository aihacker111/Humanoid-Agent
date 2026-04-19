"""
main.py — Full pipeline for Humanoid Robot Learning from Web Video

Paper: "LLM-Guided Kinematic Retargeting for Zero-Shot Humanoid Learning
        from In-the-Wild Internet Video"

Pipeline:
  YouTube Video
    → VLM Scene Understanding
    → MediaPipe Holistic Pose Extraction
    → Action Segmentation → Skill Library
    → LLM-Guided Kinematic Retargeting      ← CORE NOVELTY
    → Genesis Humanoid Simulation (H1)
    → Multi-metric Paper Evaluation

Usage:
  python main.py --source data/videos/cooking.mp4 --task "pick up cup and walk to table"
  python main.py --source "https://youtube.com/..." --task "pour water into glass"
  python main.py --task "walk to shelf and pick object" --use-library
"""
import time
import argparse
import numpy as np
from pathlib import Path
import cv2

from config import config
from agents.video_agent import VideoUnderstandingAgent
from agents.pose_agent import PoseExtractionAgent
from agents.skill_agent import SkillExtractionAgent
from agents.retargeting_agent import RetargetingAgent
from models import RetargetedSequence
from simulation.genesis_env import HumanoidEnv
from simulation.evaluator import PaperEvaluator
from core.trajectory_recorder import TrajectoryRecorder, TrajectoryPlayer
from core.side_by_side_renderer import SideBySideRenderer


def run_pipeline(
    source: str,
    task: str,
    use_library: bool = False,
    show_viewer: bool = False,
    max_frames: int = None,
):
    """
    Full end-to-end pipeline from video to simulation result.
    """
    print("\n" + "═" * 65)
    print("  Humanoid Learning from Web Video — Full Pipeline")
    print("═" * 65)
    print(f"  Source : {source}")
    print(f"  Task   : {task}")
    print("═" * 65 + "\n")

    # ── Download YouTube if needed ────────────────────────────────────────────
    if source.startswith("http"):
        print("[Pipeline] Downloading video...")
        source = _download_youtube(source)

    # ── Initialize agents ─────────────────────────────────────────────────────
    video_agent   = VideoUnderstandingAgent()
    pose_agent    = PoseExtractionAgent()
    skill_agent   = SkillExtractionAgent()
    retargeter    = RetargetingAgent()
    env           = HumanoidEnv(show_viewer=show_viewer, task=task)
    evaluator     = PaperEvaluator()

    # ── Initialize recorder & renderer ───────────────────────────────────────
    from datetime import datetime
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    recorder = TrajectoryRecorder(session_id=session_id)
    recorder.start_session(task=task, source_video=source)

    comparison_path = f"outputs/videos/{session_id}_comparison.mp4"
    renderer = SideBySideRenderer(output_path=comparison_path, task=task)

    # ── Step 1: Scene understanding ───────────────────────────────────────────
    print("Step 1 / 5 — Scene understanding")
    cap = cv2.VideoCapture(source)
    ret, first_frame = cap.read()
    cap.release()

    scene = video_agent.analyze_frame(
        first_frame if ret else _blank_frame(),
        frame_number=0,
    )
    print(f"  Scene : {scene.description}")
    print(f"  Env   : {scene.environment_type}")
    print(f"  Goal  : {scene.task_goal}")

    # ── Step 2: Pose extraction ───────────────────────────────────────────────
    print("\nStep 2 / 5 — Full-body pose extraction (MediaPipe Holistic)")
    poses = []
    frames = []
    count = 0

    for pose in pose_agent.extract_from_video(source):
        poses.append(pose)
        count += 1
        if max_frames and count >= max_frames:
            break

    print(f"  Extracted {len(poses)} poses")
    loco = sum(1 for p in poses if p.is_moving)
    print(f"  Locomotion frames : {loco} / {len(poses)}")

    if not poses:
        print("  [!] No poses extracted. Check video source.")
        return None

    # ── Step 3: Skill extraction ──────────────────────────────────────────────
    print("\nStep 3 / 5 — Action segmentation & skill library")

    if use_library and skill_agent.skill_library:
        print(f"  Using existing library: {len(skill_agent.skill_library)} skills")
        skill_sequence = skill_agent.compose_skill_sequence(task)
    else:
        primitives = skill_agent.segment_video(poses, scene)
        print(f"  Segmented {len(primitives)} primitives")

        for p in primitives:
            print(f"    [{p.control_mode.value:12s}] {p.action_type.value} "
                  f"({p.duration_seconds:.1f}s, conf={p.confidence:.2f})")
            skill_agent.add_to_library(p)

        from models import SkillSequence
        skill_sequence = SkillSequence(
            task_description=task,
            primitives=primitives,
            estimated_duration=sum(p.duration_seconds for p in primitives),
            requires_locomotion=any(
                p.control_mode.value in ["locomotion", "whole_body"]
                for p in primitives
            ),
            requires_manipulation=any(
                p.control_mode.value in ["manipulation", "whole_body"]
                for p in primitives
            ),
        )

    # ── Step 4: LLM-guided retargeting ────────────────────────────────────────
    print("\nStep 4 / 5 — LLM-guided kinematic retargeting (CORE NOVELTY)")
    print(f"  Using: {config.openrouter.reasoning_model}")
    print(f"  Retargeting {len(poses)} poses → H1 joint space ({config.humanoid.total_dof} DOF)")

    t_retarget = time.time()
    retargeted_frames = retargeter.retarget_sequence(
        poses=poses,           # All poses — keyframe selection handles efficiency
        task_context=task,
        max_keyframes=8,       # Only 8 poses sent to LLM → 1 batch API call
        batch_size=8,          # All keyframes in 1 call
    )
    retarget_time = time.time() - t_retarget

    # ── Record retargeted trajectory ──────────────────────────────────────────
    for frame in retargeted_frames:
        recorder.record_frame(frame)
    saved_paths = recorder.save()

    retargeted_sequence = RetargetedSequence(
        primitive_id="full_sequence",
        frames=retargeted_frames,
        success=len(retargeted_frames) > 0,
    )

    # Retargeting quality report
    if retargeted_frames:
        avg_balance = sum(f.balance_score for f in retargeted_frames) / len(retargeted_frames)
        avg_reach = sum(f.reachability_score for f in retargeted_frames) / len(retargeted_frames)
        avg_error = sum(f.retargeting_error for f in retargeted_frames) / len(retargeted_frames)
        kf_count = min(8, len(poses))
        api_calls = max(1, kf_count // 8)
        print(f"  API calls        : {api_calls} (batch of {kf_count} keyframes)")
        print(f"  Balance score    : {avg_balance:.3f}")
        print(f"  Reachability     : {avg_reach:.3f}")
        print(f"  Retargeting error: {avg_error:.3f}")
        print(f"  Time             : {retarget_time:.1f}s  (~{retarget_time/max(len(poses),1):.1f}s/pose)")

    # ── Step 5: Simulation & evaluation ───────────────────────────────────────
    print("\nStep 5 / 5 — Genesis simulation & paper evaluation")

    # Pass human frames so evaluator can judge naturalness (same timecodes as retargeting)
    n_eval = min(10, len(retargeted_frames))
    eval_nums = [
        retargeted_frames[i].source_human_pose.frame_number
        for i in range(n_eval)
    ]
    human_frames_for_eval = _extract_frames_by_frame_numbers(source, eval_nums)
    evaluator.set_human_frames(human_frames_for_eval)

    result = evaluator.evaluate_skill_sequence(
        skill_sequence=skill_sequence,
        retargeted=retargeted_sequence,
        env=env,
        source_video=source,
    )

    sim_frames = evaluator.get_last_sim_frames()

    # ── Generate side-by-side comparison video ────────────────────────────────
    print("\n[Pipeline] Generating side-by-side comparison video...")
    renderer.add_title_card(
        title=f"Human Demo  vs  {config.humanoid.display_name}",
        subtitle=f"Task: {task}",
        duration_seconds=1.5,
    )

    # Human pixels aligned with pose extraction frame indices (not uniform subsampling)
    frame_nums = [f.source_human_pose.frame_number for f in retargeted_frames]
    human_frames_raw = _extract_frames_by_frame_numbers(source, frame_nums)
    renderer.set_total_frames(len(retargeted_frames))

    for i, (h_frame, r_pose) in enumerate(zip(human_frames_raw, retargeted_frames)):
        human_pose = poses[i] if i < len(poses) else None
        if i < len(sim_frames) and sim_frames[i] is not None and sim_frames[i].size > 0:
            robot_frame = sim_frames[i]
        else:
            robot_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        renderer.write_frame(
            human_frame=h_frame,
            robot_frame=robot_frame,
            human_pose=human_pose,
            retargeted_pose=r_pose,
            extra_metrics={
                "naturalness": round(result.motion_naturalness, 2),
                "stability":   round(result.gait_stability, 2),
            },
        )

    video_path = renderer.finalize()

    # ── Save results ──────────────────────────────────────────────────────────
    evaluator.save_results()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  PAPER METRICS SUMMARY")
    print("═" * 65)
    table = evaluator.compute_paper_table()
    for k, v in table.items():
        if k != "per_task":
            print(f"  {k:<30}: {v}")

    print("\n  OUTPUT FILES")
    print("  " + "─" * 40)
    for name, path in saved_paths.items():
        print(f"  [{name:<20}] {path}")
    print(f"  [{'comparison_video':<20}] {video_path}")
    print(f"  [{'eval_results':<20}] outputs/evaluation_results.json")
    print("═" * 65)

    # Explicitly close MediaPipe to avoid __del__ warnings
    if hasattr(pose_agent, 'close'):
        pose_agent.close()

    env.close()
    return result


# ── Benchmark tasks for paper ─────────────────────────────────────────────────

BENCHMARK_TASKS = [
    # Locomotion only
    "walk forward to the table",
    "turn left and walk to the door",
    # Manipulation only
    "pick up the cup from the table",
    "pour water from bottle into glass",
    # Whole-body (loco + manip simultaneously)
    "walk to shelf and pick up the box",
    "carry the object while walking",
    "open the door while walking through",
]


def run_benchmark(source: str, show_viewer: bool = False):
    """
    Run full benchmark suite for paper Table 1.
    """
    print(f"\nRunning benchmark on {len(BENCHMARK_TASKS)} tasks...")
    results = []
    for task in BENCHMARK_TASKS:
        result = run_pipeline(
            source=source,
            task=task,
            show_viewer=show_viewer,
            max_frames=30,
        )
        if result:
            results.append(result)
    print(f"\nBenchmark complete: {len(results)} tasks evaluated")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_frames_for_render(video_path: str, n: int) -> list:
    """Evenly subsample frames (legacy helper; prefer _extract_frames_by_frame_numbers)."""
    import numpy as np
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [np.zeros((480, 640, 3), dtype=np.uint8)] * n
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max(n, 1))
    frames, count = [], 0
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % step == 0:
            frames.append(frame)
    cap.release()
    if frames and len(frames) < n:
        frames += [frames[-1]] * (n - len(frames))
    return frames[:n]


def _extract_frames_by_frame_numbers(video_path: str, frame_numbers: list) -> list:
    """
    Grab specific frames by 1-based index (matches PoseExtractionAgent frame_number).
    Required so skeleton overlay matches the HumanPose used for each retargeted frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [np.zeros((480, 640, 3), dtype=np.uint8)] * len(frame_numbers)

    out = []
    fallback = None
    for fn in frame_numbers:
        idx = max(0, int(fn) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            fallback = frame
            out.append(frame)
        elif fallback is not None:
            out.append(fallback.copy())
        else:
            out.append(np.zeros((480, 640, 3), dtype=np.uint8))
    cap.release()
    return out


def _download_youtube(url: str) -> str:
    try:
        import yt_dlp
        output = "data/videos/downloaded.mp4"
        Path("data/videos").mkdir(parents=True, exist_ok=True)
        ydl_opts = {"format": "best[height<=720]", "outtmpl": output, "quiet": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output
    except ImportError:
        raise ImportError("Install yt-dlp: pip install yt-dlp")


def _blank_frame():
    import numpy as np
    return np.zeros((480, 640, 3), dtype=np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Humanoid Robot Learning from Web Video"
    )
    parser.add_argument("--source", default="data/videos/demo.mp4",
                        help="Video path or YouTube URL")
    parser.add_argument("--task", default="walk to table and pick up cup",
                        help="Task description in natural language")
    parser.add_argument("--use-library", action="store_true",
                        help="Use existing skill library instead of relearning")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full benchmark suite for paper")
    parser.add_argument("--show-viewer", action="store_true",
                        help="Open Genesis interactive 3D viewer (needs a display)")
    parser.add_argument("--no-offscreen-camera", action="store_true",
                        help="Disable headless 3D camera; comparison video uses 2D HUD fallback")
    parser.add_argument("--max-frames", type=int, default=50,
                        help="Max frames to process (for quick testing)")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    config.debug = args.debug
    if args.no_offscreen_camera:
        config.simulation.offscreen_camera = False

    if args.benchmark:
        run_benchmark(args.source, show_viewer=args.show_viewer)
    else:
        run_pipeline(
            source=args.source,
            task=args.task,
            use_library=args.use_library,
            show_viewer=args.show_viewer,
            max_frames=args.max_frames,
        )
