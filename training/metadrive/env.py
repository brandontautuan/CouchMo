"""CouchMoMetaDriveEnv — gym wrapper around SafeMetaDriveEnv.

Produces stereo-cam observations shaped (8, 84, 84) float32 in [0,1] by running
MetaDrive's two attached RGBCameras through shared.preprocess.preprocess_pair
and shared.preprocess.FrameStacker. Action space matches the serial protocol
exactly: Box(low=[-1, 0], high=[1, 1]).

This initial version covers obs and action wiring only. Reward, termination,
and domain randomization land in later tasks.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover — heavy optional dep
    raise ImportError(
        "gymnasium is required; pip install -r training/requirements-rl.txt"
    ) from exc

try:
    from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
except ImportError as exc:  # pragma: no cover — heavy optional dep
    raise ImportError(
        "metadrive is required; pip install -r training/requirements-rl.txt"
    ) from exc

from shared.preprocess import FrameStacker, preprocess_pair

# Cam geometry mimics the real Brio 100 mounts (see autonomous_car_research.md).
CAM_LATERAL_M: float = 0.35
CAM_HEIGHT_M: float = 0.81
CAM_PITCH_DEG: float = -5.0     # mild downward tilt
CAM_WIDTH: int = 256
CAM_HEIGHT: int = 256
DECISION_REPEAT: int = 10       # MetaDrive physics @ 100 Hz; policy @ 10 Hz


class CouchMoMetaDriveEnv(gym.Env):
    """Gym env that wraps SafeMetaDriveEnv with stereo cams + CouchMo action space."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()

        cfg: dict[str, Any] = {
            "num_scenarios": 1000,
            "start_seed": 0,
            "traffic_density": 0.1,
            "accident_prob": 0.0,
            "use_render": False,
            "manual_control": False,
            "decision_repeat": DECISION_REPEAT,
            "sensors": {
                "left_cam": (RGBCamera, CAM_WIDTH, CAM_HEIGHT),
                "right_cam": (RGBCamera, CAM_WIDTH, CAM_HEIGHT),
            },
            "vehicle_config": {
                "image_source": "left_cam",  # default — we read both manually
            },
        }
        if config:
            cfg.update(config)

        self._md_env = SafeMetaDriveEnv(cfg)
        self._stacker = FrameStacker(num_frames=4)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8, 84, 84), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        md_obs, md_info = self._md_env.reset(seed=seed)
        self._stacker.reset()
        self._attach_cameras_to_ego()
        obs = self._build_observation()
        return obs, dict(md_info)

    def step(self, action: np.ndarray):
        md_action = self._to_metadrive_action(action)
        _, reward, terminated, truncated, info = self._md_env.step(md_action)
        obs = self._build_observation()
        return obs, float(reward), bool(terminated), bool(truncated), dict(info)

    def close(self) -> None:
        self._md_env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _attach_cameras_to_ego(self) -> None:
        """Position the two RGB cameras at armrest offsets on the ego vehicle.

        MetaDrive's sensor system ties cameras to the ego after reset. This is
        called once per reset; later tasks will add per-reset jitter here.
        """
        engine = self._md_env.engine
        vehicle = engine.agent_manager.active_agents["default_agent"]
        left_cam = engine.get_sensor("left_cam")
        right_cam = engine.get_sensor("right_cam")
        # MetaDrive cams parent to the vehicle; we set local position + HPR.
        left_cam.get_cam().setPos(vehicle.origin, -CAM_LATERAL_M, 0.3, CAM_HEIGHT_M)
        right_cam.get_cam().setPos(vehicle.origin, +CAM_LATERAL_M, 0.3, CAM_HEIGHT_M)
        left_cam.get_cam().setHpr(0, CAM_PITCH_DEG, 0)
        right_cam.get_cam().setHpr(0, CAM_PITCH_DEG, 0)

    def _read_camera_rgb(self, name: str) -> np.ndarray:
        """Read a MetaDrive RGBCamera and return a (H, W, 3) uint8 BGR array.

        MetaDrive RGBCameras return RGB float in [0,1]; we convert to uint8 BGR
        to match preprocess_pair's contract (which expects BGR from cv2/OpenCV).
        """
        cam = self._md_env.engine.get_sensor(name)
        rgb = cam.perceive()  # (H, W, 3) float32 in [0, 1], RGB
        u8 = (rgb * 255.0).astype(np.uint8)
        # RGB -> BGR (reverse last axis)
        return u8[:, :, ::-1].copy()

    def _build_observation(self) -> np.ndarray:
        left_bgr = self._read_camera_rgb("left_cam")
        right_bgr = self._read_camera_rgb("right_cam")
        pair = preprocess_pair(left_bgr, right_bgr)  # (2, 84, 84) uint8
        return self._stacker.push(pair)              # (8, 84, 84) float32

    def _to_metadrive_action(self, action: np.ndarray) -> np.ndarray:
        """Convert policy action [steer, throttle] -> MetaDrive [steering, throttle_brake].

        steer is passthrough. throttle >= 0 means no brake; we pass throttle
        directly to MetaDrive's throttle_brake (which also accepts negative
        values for braking, which we never use).
        """
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        return np.array([steer, throttle], dtype=np.float32)
