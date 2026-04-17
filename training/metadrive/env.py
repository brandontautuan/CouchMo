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
CAM_FORWARD_M: float = 0.3      # forward offset from vehicle origin
CAM_PITCH_DEG: float = -5.0     # mild downward tilt
CAM_WIDTH: int = 256
CAM_HEIGHT: int = 256
DECISION_REPEAT: int = 10       # MetaDrive physics @ 100 Hz; policy @ 10 Hz

# Reward weights (see spec §Reward).
W_PROGRESS: float = 1.0
W_COLLISION: float = 50.0
W_OFF_ROAD: float = 20.0
W_SMOOTH: float = 0.1
W_IDLE: float = 0.05
IDLE_VEL_THRESHOLD: float = 0.1     # m/s
MAX_EPISODE_STEPS: int = 500

# Domain randomization ranges (see spec §Domain randomization).
STEER_GAIN_RANGE: tuple[float, float] = (0.85, 1.15)
BRIGHTNESS_RANGE: tuple[float, float] = (0.7, 1.3)
CAM_PITCH_JITTER_DEG: float = 3.0
CAM_LATERAL_JITTER_M: float = 0.02
CAM_NOISE_STD: float = 3.0          # uint8 pixel units
ACTION_DELAY_PROB: float = 0.1


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
        self._prev_action: np.ndarray = np.zeros(2, dtype=np.float32)
        self._prev_pos: np.ndarray | None = None
        self._step_count: int = 0
        self._randomize: bool = bool(cfg.get("randomize", False))
        self._rng = np.random.default_rng(0)

        # Randomization state — reset per episode.
        self._steering_gain: float = 1.0
        self._brightness_scale: float = 1.0
        self._cam_pitch_offsets = (0.0, 0.0)   # (left_deg, right_deg)
        self._cam_lateral_offsets = (0.0, 0.0) # (left_m, right_m)

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
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        md_obs, md_info = self._md_env.reset(seed=seed)
        self._stacker.reset()
        self._sample_episode_randomization()
        self._attach_cameras_to_ego()
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_pos = self._current_xy()
        self._step_count = 0
        obs = self._build_observation()
        return obs, dict(md_info)

    def step(self, action: np.ndarray):
        md_action = self._to_metadrive_action(action)
        # MetaDrive's native reward is discarded; we emit our own shaped reward below.
        _, _md_reward, md_terminated, md_truncated, info = self._md_env.step(md_action)

        self._step_count += 1
        reward, terminated = self._compute_reward_and_termination(action)
        truncated = md_truncated or self._step_count >= MAX_EPISODE_STEPS

        self._prev_action = action.astype(np.float32, copy=True)
        self._prev_pos = self._current_xy()

        obs = self._build_observation()
        return obs, float(reward), bool(terminated or md_terminated), bool(truncated), dict(info)

    def close(self) -> None:
        self._md_env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _attach_cameras_to_ego(self) -> None:
        engine = self._md_env.engine
        vehicle = engine.agent_manager.active_agents["default_agent"]
        left_cam = engine.get_sensor("left_cam")
        right_cam = engine.get_sensor("right_cam")

        left_pitch = CAM_PITCH_DEG + self._cam_pitch_offsets[0]
        right_pitch = CAM_PITCH_DEG + self._cam_pitch_offsets[1]
        left_lat = -CAM_LATERAL_M + self._cam_lateral_offsets[0]
        right_lat = +CAM_LATERAL_M + self._cam_lateral_offsets[1]

        left_cam.get_cam().setPos(vehicle.origin, left_lat, CAM_FORWARD_M, CAM_HEIGHT_M)
        right_cam.get_cam().setPos(vehicle.origin, right_lat, CAM_FORWARD_M, CAM_HEIGHT_M)
        left_cam.get_cam().setHpr(0, left_pitch, 0)
        right_cam.get_cam().setHpr(0, right_pitch, 0)

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
        left_bgr = self._apply_visual_randomization(self._read_camera_rgb("left_cam"))
        right_bgr = self._apply_visual_randomization(self._read_camera_rgb("right_cam"))
        pair = preprocess_pair(left_bgr, right_bgr)
        return self._stacker.push(pair)

    def _current_xy(self) -> np.ndarray:
        vehicle = self._md_env.engine.agent_manager.active_agents["default_agent"]
        pos = vehicle.position  # (x, y)
        return np.array([pos[0], pos[1]], dtype=np.float32)

    def _current_speed(self) -> float:
        vehicle = self._md_env.engine.agent_manager.active_agents["default_agent"]
        return float(vehicle.speed)  # m/s

    def _is_collision(self) -> bool:
        vehicle = self._md_env.engine.agent_manager.active_agents["default_agent"]
        return bool(
            getattr(vehicle, "crash_vehicle", False)
            or getattr(vehicle, "crash_object", False)
            or getattr(vehicle, "crash_sidewalk", False)
        )

    def _is_off_road(self) -> bool:
        vehicle = self._md_env.engine.agent_manager.active_agents["default_agent"]
        return bool(getattr(vehicle, "out_of_road", False))

    def _compute_reward_and_termination(
        self, action: np.ndarray
    ) -> tuple[float, bool]:
        # _prev_pos is always seeded by reset(), so no None-guard is needed:
        # calling step() before reset() is a gym API violation and should fail loud.
        reward = W_PROGRESS * float(np.linalg.norm(self._current_xy() - self._prev_pos))

        # Action smoothness (quadratic penalty on action delta from previous step).
        delta = action.astype(np.float32) - self._prev_action
        reward -= W_SMOOTH * float(np.dot(delta, delta))

        # Idle penalty: always fire when speed is below the idle threshold —
        # the agent must learn to keep moving.
        if self._current_speed() < IDLE_VEL_THRESHOLD:
            reward -= W_IDLE

        # Terminal rewards. Collision is strictly worse than off-road, so it wins.
        terminated = False
        if self._is_collision():
            reward -= W_COLLISION
            terminated = True
        elif self._is_off_road():
            reward -= W_OFF_ROAD
            terminated = True

        return reward, terminated

    def _sample_episode_randomization(self) -> None:
        if not self._randomize:
            self._steering_gain = 1.0
            self._brightness_scale = 1.0
            self._cam_pitch_offsets = (0.0, 0.0)
            self._cam_lateral_offsets = (0.0, 0.0)
            return

        self._steering_gain = float(self._rng.uniform(*STEER_GAIN_RANGE))
        self._brightness_scale = float(self._rng.uniform(*BRIGHTNESS_RANGE))
        self._cam_pitch_offsets = (
            float(self._rng.uniform(-CAM_PITCH_JITTER_DEG, CAM_PITCH_JITTER_DEG)),
            float(self._rng.uniform(-CAM_PITCH_JITTER_DEG, CAM_PITCH_JITTER_DEG)),
        )
        self._cam_lateral_offsets = (
            float(self._rng.uniform(-CAM_LATERAL_JITTER_M, CAM_LATERAL_JITTER_M)),
            float(self._rng.uniform(-CAM_LATERAL_JITTER_M, CAM_LATERAL_JITTER_M)),
        )

    def _apply_visual_randomization(self, bgr: np.ndarray) -> np.ndarray:
        """Apply brightness scale + Gaussian noise to a uint8 BGR frame."""
        if not self._randomize:
            return bgr
        scaled = bgr.astype(np.float32) * self._brightness_scale
        noise = self._rng.normal(0.0, CAM_NOISE_STD, size=bgr.shape).astype(np.float32)
        noisy = np.clip(scaled + noise, 0.0, 255.0)
        return noisy.astype(np.uint8)

    def _to_metadrive_action(self, action: np.ndarray) -> np.ndarray:
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))

        if self._randomize:
            # Steering gain (bridges skid-steer dynamics gap).
            steer *= self._steering_gain

            # Action delay — occasionally re-apply the previous raw policy
            # command (with the current episode's gain). Replaying the raw
            # command, not the previously-sent post-gain value, keeps
            # action-delay and steering-gain as independent randomization
            # knobs rather than compounding gain across repeats.
            if self._rng.random() < ACTION_DELAY_PROB:
                steer = float(self._prev_action[0]) * self._steering_gain
                throttle = float(self._prev_action[1])

            # Single clip covers both the gain-only and action-delay branches.
            steer = float(np.clip(steer, -1.0, 1.0))

        return np.array([steer, throttle], dtype=np.float32)
