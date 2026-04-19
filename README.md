# LLM-Guided Kinematic Retargeting for Humanoid Robot Learning

> **Zero-shot humanoid learning from in-the-wild internet video**
> No teleoperation. No motion capture. No robot training required.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

Get your API key at: https://openrouter.ai/keys

## Run

```bash
# Single task
python main.py --source your_video.mp4 --task "bimanual hand rubbing gesture while standing"

# With YouTube
python main.py --source "https://youtube.com/watch?v=..." --task "walk and pick up cup"

# Full paper benchmark
python main.py --benchmark --source your_video.mp4
```

## Project Structure

```
humanoid_agent/
├── config.py                        # Config + auto load_dotenv
├── main.py                          # Pipeline entry point
├── .env.example                     # Copy to .env and add API key
├── agents/
│   ├── video_agent.py               # VLM scene understanding (Qwen2.5-VL)
│   ├── pose_agent.py                # MediaPipe Tasks API (0.10.x+)
│   ├── retargeting_agent.py         # LLM retargeting ← CORE NOVELTY
│   └── skill_agent.py               # Action segmentation + skill library
├── core/
│   ├── openrouter.py                # API client with robust JSON parsing
│   ├── trajectory_recorder.py       # Save trajectories + reasoning logs
│   └── side_by_side_renderer.py     # Human vs Robot comparison video
├── simulation/
│   ├── genesis_env.py               # Genesis + H1 auto-download
│   └── evaluator.py                 # Paper evaluation metrics
├── models/__init__.py               # All Pydantic schemas
└── outputs/                         # All saved results
    ├── trajectories/                # Joint angle data (.json + .npy)
    ├── reasoning/                   # LLM chain-of-thought logs (.md + .json)
    └── videos/                      # Side-by-side comparison videos
```

## Fixes Applied

| Bug | Fix |
|---|---|
| `mp.solutions.holistic` removed | Rewrote with MediaPipe Tasks API |
| `urdf_path` attribute missing | Removed from config, genesis_env auto-resolves |
| Genesis file not found | `_ensure_h1_model()` tries 4 strategies |
| `stand_manipulate` invalid ActionType | `_safe_action_type()` maps unknown values |
| LLM returns empty/extra-text JSON | `_extract_json_safe()` with brace matching |
| 401 Unauthorized | `load_dotenv()` auto-called in config.py |
| `show_viewer=True` crashes Colab | Default changed to `False` |

## Paper Metrics

The `/stats` endpoint and `compute_paper_table()` return:

| Metric | Description |
|---|---|
| `success_rate` | Task completion % |
| `avg_gait_stability` | COM stability during locomotion |
| `avg_motion_naturalness` | VLM judge score (0–1) |
| `avg_coordination_score` | Loco-manip sync quality |
| `avg_fall_count` | Falls per episode |
