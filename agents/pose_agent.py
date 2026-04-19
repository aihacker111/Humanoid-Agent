"""
agents/pose_agent.py — Full-body pose extraction
MediaPipe Tasks API 0.10.x+ (mp.solutions removed)
Auto-downloads models on first run. Falls back to mock if offline.
"""
import os
import cv2
import urllib.request
import numpy as np
from pathlib import Path
from typing import Generator, Optional

from models import HumanPose, Keypoint3D, ControlMode
from config import config


MODEL_DIR = Path("assets/models")
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_heavy.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

BODY_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

HAND_LANDMARK_NAMES = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def _download_model(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 10_000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  Downloading {dest.name} ...")
        urllib.request.urlretrieve(url, str(dest))
        if dest.stat().st_size > 10_000:
            print(f"  OK ({dest.stat().st_size // 1024} KB)")
            return True
        dest.unlink(missing_ok=True)
        return False
    except Exception as e:
        print(f"  Download failed: {e}")
        dest.unlink(missing_ok=True)
        return False


class PoseExtractionAgent:
    def __init__(self):
        self.sample_rate = config.pose.sample_rate
        self._pose_lm = None
        self._hand_lm = None
        self._mock_mode = False
        self._prev_hip = None
        self._setup()

    def _setup(self):
        print("[PoseAgent] Initializing MediaPipe Tasks API ...")
        pose_ok = _download_model(POSE_MODEL_URL, POSE_MODEL_PATH)
        hand_ok = _download_model(HAND_MODEL_URL, HAND_MODEL_PATH)

        if not pose_ok:
            print("[PoseAgent] Models unavailable — MOCK mode")
            self._mock_mode = True
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision as mpv
            from mediapipe.tasks.python.core.base_options import BaseOptions

            self._pose_lm = mpv.PoseLandmarker.create_from_options(
                mpv.PoseLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(POSE_MODEL_PATH)
                    ),
                    running_mode=mpv.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=config.pose.min_detection_confidence,
                    min_pose_presence_confidence=config.pose.min_tracking_confidence,
                    min_tracking_confidence=config.pose.min_tracking_confidence,
                    output_segmentation_masks=False,
                )
            )

            if hand_ok:
                self._hand_lm = mpv.HandLandmarker.create_from_options(
                    mpv.HandLandmarkerOptions(
                        base_options=BaseOptions(
                            model_asset_path=str(HAND_MODEL_PATH)
                        ),
                        running_mode=mpv.RunningMode.IMAGE,
                        num_hands=2,
                        min_hand_detection_confidence=0.5,
                        min_hand_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                )

            print("[PoseAgent] Ready — PoseLandmarker + HandLandmarker")

        except Exception as e:
            print(f"[PoseAgent] Setup error: {e} — MOCK mode")
            self._mock_mode = True

    def extract_from_video(
        self, video_path: str
    ) -> Generator[HumanPose, None, None]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0

        print(
            f"[PoseAgent] {Path(video_path).name} — "
            f"{fps:.0f}fps, {total} frames, sample every {self.sample_rate}"
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                count += 1
                if count % self.sample_rate != 0:
                    continue
                pose = self._process_frame(frame, count, count / fps)
                if pose:
                    yield pose
        finally:
            cap.release()

    def extract_from_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
    ) -> Optional[HumanPose]:
        return self._process_frame(frame, frame_number, timestamp)

    def draw_pose(self, frame: np.ndarray, pose: HumanPose) -> np.ndarray:
        out = frame.copy()
        h, w = frame.shape[:2]
        for kp in pose.body.values():
            if kp.visibility > 0.5:
                cv2.circle(out, (int(kp.x * w), int(kp.y * h)), 4, (50, 205, 50), -1)
        for kp in list(pose.left_hand.values()) + list(pose.right_hand.values()):
            cv2.circle(out, (int(kp.x * w), int(kp.y * h)), 2, (255, 200, 0), -1)
        c = {
            ControlMode.LOCOMOTION:   (255, 128, 0),
            ControlMode.MANIPULATION: (0, 180, 255),
            ControlMode.WHOLE_BODY:   (180, 0, 255),
        }.get(pose.dominant_mode, (200, 200, 200))
        cv2.putText(out, pose.dominant_mode.value, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        return out

    def _process_frame(
        self, frame: np.ndarray, frame_number: int, timestamp: float
    ) -> Optional[HumanPose]:
        if self._mock_mode:
            return self._make_mock(frame_number, timestamp)
        try:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            pr = self._pose_lm.detect(mp_img)
            if not pr.pose_landmarks:
                return None

            body = {
                BODY_LANDMARK_NAMES[i]: Keypoint3D(
                    x=float(lm.x), y=float(lm.y), z=float(lm.z),
                    visibility=float(getattr(lm, "visibility", 1.0)),
                )
                for i, lm in enumerate(pr.pose_landmarks[0])
                if i < len(BODY_LANDMARK_NAMES)
            }

            left_hand, right_hand = {}, {}
            if self._hand_lm:
                hr = self._hand_lm.detect(mp_img)
                for hi, hlms in enumerate(hr.hand_landmarks):
                    side = (
                        hr.handedness[hi][0].category_name
                        if hr.handedness else "Left"
                    )
                    target = left_hand if side == "Left" else right_hand
                    for i, lm in enumerate(hlms):
                        if i < len(HAND_LANDMARK_NAMES):
                            target[HAND_LANDMARK_NAMES[i]] = Keypoint3D(
                                x=float(lm.x), y=float(lm.y), z=float(lm.z)
                            )

            com = self._com(body)
            moving = self._moving(body)
            has_hands = bool(left_hand or right_hand)

            if moving and has_hands:
                mode = ControlMode.WHOLE_BODY
            elif moving:
                mode = ControlMode.LOCOMOTION
            else:
                mode = ControlMode.MANIPULATION

            return HumanPose(
                frame_number=frame_number, timestamp=timestamp,
                body=body, left_hand=left_hand, right_hand=right_hand,
                is_moving=moving, center_of_mass=com, dominant_mode=mode,
            )
        except Exception as e:
            if config.debug:
                print(f"[PoseAgent] frame {frame_number}: {e}")
            return self._make_mock(frame_number, timestamp)

    def _com(self, body: dict) -> Optional[Keypoint3D]:
        l, r = body.get("left_hip"), body.get("right_hip")
        if not l or not r:
            return None
        return Keypoint3D(x=(l.x+r.x)/2, y=(l.y+r.y)/2, z=(l.z+r.z)/2)

    def _moving(self, body: dict) -> bool:
        l, r = body.get("left_hip"), body.get("right_hip")
        if not l or not r:
            return False
        cx, cy = (l.x+r.x)/2, (l.y+r.y)/2
        if self._prev_hip is None:
            self._prev_hip = (cx, cy)
            return False
        d = abs(cx - self._prev_hip[0]) + abs(cy - self._prev_hip[1])
        self._prev_hip = (cx, cy)
        return d > 0.01

    def _make_mock(self, frame_number: int, timestamp: float) -> HumanPose:
        t = timestamp * 0.5
        body = {
            name: Keypoint3D(
                x=0.5 + 0.04 * np.sin(t + i * 0.3),
                y=float(i) / len(BODY_LANDMARK_NAMES),
                z=0.0, visibility=0.9,
            )
            for i, name in enumerate(BODY_LANDMARK_NAMES)
        }
        lh = {
            name: Keypoint3D(
                x=0.35 + 0.02 * np.sin(t * 2 + i),
                y=0.55 + 0.02 * np.cos(t * 2 + i), z=0.0,
            )
            for i, name in enumerate(HAND_LANDMARK_NAMES)
        }
        rh = {
            name: Keypoint3D(
                x=0.65 + 0.02 * np.sin(t * 2 + i),
                y=0.55 + 0.02 * np.cos(t * 2 + i), z=0.0,
            )
            for i, name in enumerate(HAND_LANDMARK_NAMES)
        }
        return HumanPose(
            frame_number=frame_number, timestamp=timestamp,
            body=body, left_hand=lh, right_hand=rh,
            is_moving=False,
            center_of_mass=Keypoint3D(x=0.5, y=0.5, z=0.0),
            dominant_mode=ControlMode.MANIPULATION,
        )

    def close(self):
        """Explicitly close landmarkers — call this before program exit."""
        for lm in [self._pose_lm, self._hand_lm]:
            if lm:
                try:
                    lm.close()
                except Exception:
                    pass
        self._pose_lm = None
        self._hand_lm = None

    def __del__(self):
        # Suppress mediapipe cleanup errors at interpreter shutdown
        # This is a known mediapipe bug — safe to ignore
        try:
            self.close()
        except Exception:
            pass
