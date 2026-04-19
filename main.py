"""
main.py — Full pipeline for Humanoid Robot Learning from Web Video
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
from core.side_by_side_renderer import SideBySideRenderer, create_comparison_video


def run_pipeline(source, task, use_library=False, show_viewer=False, max_frames=None):
    print("\n" + "═"*65)
    print("  Humanoid Learning from Web Video — Full Pipeline")
    print("═"*65)
    print(f"  Source : {source}")
    print(f"  Task   : {task}")
    print("═"*65 + "\n")

    if source.startswith("http"):
        print("[Pipeline] Downloading video...")
        source = _download_youtube(source)

    video_agent  = VideoUnderstandingAgent()
    pose_agent   = PoseExtractionAgent()
    skill_agent  = SkillExtractionAgent()
    retargeter   = RetargetingAgent()
    env          = HumanoidEnv(show_viewer=show_viewer, task=task)
    evaluator    = PaperEvaluator()

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
    scene = video_agent.analyze_frame(first_frame if ret else _blank_frame(), frame_number=0)
    print(f"  Scene : {scene.description}")
    print(f"  Env   : {scene.environment_type}")
    print(f"  Goal  : {scene.task_goal}")

    # ── Step 2: Pose extraction ───────────────────────────────────────────────
    print("\nStep 2 / 5 — Full-body pose extraction (MediaPipe)")
    poses = []
    count = 0
    for pose in pose_agent.extract_from_video(source):
        poses.append(pose)
        count += 1
        if max_frames and count >= max_frames:
            break
    print(f"  Extracted {len(poses)} poses")
    print(f"  Locomotion frames : {sum(1 for p in poses if p.is_moving)} / {len(poses)}")
    if not poses:
        print("  [!] No poses extracted.")
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
            requires_locomotion=any(p.control_mode.value in ["locomotion","whole_body"] for p in primitives),
            requires_manipulation=any(p.control_mode.value in ["manipulation","whole_body"] for p in primitives),
        )

    # ── Step 4: LLM-guided retargeting ────────────────────────────────────────
    print("\nStep 4 / 5 — LLM-guided kinematic retargeting (CORE NOVELTY)")
    print(f"  Using: {config.openrouter.reasoning_model}")
    print(f"  Retargeting {len(poses)} poses → H1 joint space ({config.humanoid.total_dof} DOF)")
    t0 = time.time()
    retargeted_frames = retargeter.retarget_sequence(
        poses=poses, task_context=task, max_keyframes=8, batch_size=8,
    )
    t_ret = time.time() - t0
    for frame in retargeted_frames:
        recorder.record_frame(frame)
    saved_paths = recorder.save()
    retargeted_sequence = RetargetedSequence(
        primitive_id="full_sequence",
        frames=retargeted_frames,
        success=len(retargeted_frames) > 0,
    )
    if retargeted_frames:
        print(f"  API calls        : 1 (batch of {min(8, len(poses))} keyframes)")
        print(f"  Balance score    : {sum(f.balance_score for f in retargeted_frames)/len(retargeted_frames):.3f}")
        print(f"  Reachability     : {sum(f.reachability_score for f in retargeted_frames)/len(retargeted_frames):.3f}")
        print(f"  Time             : {t_ret:.1f}s")

    # ── Step 5: Simulation & evaluation ───────────────────────────────────────
    print("\nStep 5 / 5 — Genesis simulation & paper evaluation")
    human_frames_for_eval = _extract_frames_for_render(source, min(10, len(retargeted_frames)))
    evaluator.set_human_frames(human_frames_for_eval)
    result = evaluator.evaluate_skill_sequence(
        skill_sequence=skill_sequence,
        retargeted=retargeted_sequence,
        env=env,
        source_video=source,
    )

    # ── Generate comparison video ─────────────────────────────────────────────
    print("\n[Pipeline] Generating side-by-side comparison video...")
    renderer.add_title_card(
        title="Human Demo  vs  Unitree H1",
        subtitle=f"Task: {task}",
        duration_seconds=1.5,
    )
    human_frames_raw = _extract_frames_for_render(source, len(retargeted_frames))

    # ── FIX: Re-run sim với render_frames=True để lấy ảnh 3D thật từ Genesis ──
    print("[Pipeline] Rendering Genesis 3D frames...")
    env.reset()
    trajectory = [f.joint_angles for f in retargeted_frames]
    robot_frames_3d = env.execute_sequence(trajectory, render_frames=True)
    n_real = sum(1 for f in robot_frames_3d if f is not None)
    print(f"[Pipeline] {n_real}/{len(robot_frames_3d)} real Genesis 3D frames")

    for i, (h_frame, r_pose) in enumerate(zip(human_frames_raw, retargeted_frames)):
        human_pose = poses[i] if i < len(poses) else None
        # ── FIX: Dùng frame 3D thật, không dùng np.zeros ──────────────────
        robot_frame = robot_frames_3d[i] if i < len(robot_frames_3d) else None
        renderer.write_frame(
            human_frame=h_frame,
            robot_frame=robot_frame,
            human_pose=human_pose,
            retargeted_pose=r_pose,
            # ── FIX: metrics= thay vì extra_info= ─────────────────────────
            metrics={
                "naturalness":  round(result.motion_naturalness, 2),
                "balance":      round(result.gait_stability, 2),
                "coordination": round(result.coordination_score, 2) if hasattr(result, "coordination_score") else 0.0,
                "fall_count":   result.fall_count,
            },
        )

    video_path = renderer.finalize()
    evaluator.save_results()

    print("\n" + "═"*65)
    print("  PAPER METRICS SUMMARY")
    print("═"*65)
    table = evaluator.compute_paper_table()
    for k, v in table.items():
        if k != "per_task":
            print(f"  {k:<30}: {v}")
    print("\n  OUTPUT FILES")
    print("  " + "─"*40)
    for name, path in saved_paths.items():
        print(f"  [{name:<20}] {path}")
    print(f"  [{'comparison_video':<20}] {video_path}")
    print(f"  [{'eval_results':<20}] outputs/evaluation_results.json")
    print("═"*65)

    if hasattr(pose_agent, 'close'):
        pose_agent.close()
    env.close()
    return result


BENCHMARK_TASKS = [
    "walk forward to the table",
    "turn left and walk to the door",
    "pick up the cup from the table",
    "pour water from bottle into glass",
    "walk to shelf and pick up the box",
    "carry the object while walking",
    "open the door while walking through",
]


def run_benchmark(source, show_viewer=False):
    print(f"\nRunning benchmark on {len(BENCHMARK_TASKS)} tasks...")
    results = []
    for task in BENCHMARK_TASKS:
        result = run_pipeline(source=source, task=task,
                              show_viewer=show_viewer, max_frames=30)
        if result:
            results.append(result)
    print(f"\nBenchmark complete: {len(results)} tasks evaluated")


def _extract_frames_for_render(video_path, n):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [None] * n
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max(n, 1))
    frames, count = [], 0
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret: break
        count += 1
        if count % step == 0:
            frames.append(frame)
    cap.release()
    if frames and len(frames) < n:
        frames += [frames[-1]] * (n - len(frames))
    return frames[:n] if frames else [None] * n


def _download_youtube(url):
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
    return np.zeros((480, 640, 3), dtype=np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",      default="data/videos/demo.mp4")
    parser.add_argument("--task",        default="walk to table and pick up cup")
    parser.add_argument("--use-library", action="store_true")
    parser.add_argument("--benchmark",   action="store_true")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--max-frames",  type=int, default=50)
    parser.add_argument("--debug",       action="store_true")
    args = parser.parse_args()
    config.debug = args.debug
    if args.benchmark:
        run_benchmark(args.source, show_viewer=args.show_viewer)
    else:
        run_pipeline(
            source=args.source, task=args.task,
            use_library=args.use_library, show_viewer=args.show_viewer,
            max_frames=args.max_frames,
        )