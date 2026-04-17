# MetaDrive RL Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]` / `- [x]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-04-17-metadrive-rl-training-design.md`

**Goal:** Add a MetaDrive-based RL training pipeline that pretrains `BCPolicy` via behavior cloning on MetaDrive's IDM expert, fine-tunes with PPO under domain randomization, and exports an ONNX artifact that `runtime/inference.py` can load with zero changes.

**Architecture:** New sibling directory `training/metadrive/` hosts a Gym-compatible `CouchMoMetaDriveEnv` wrapper (stereo cam obs → `(8, 84, 84)`, `(steer, throttle)` action), an IDM expert adapter, a BC data-collection CLI, a custom SB3 `ActorCriticPolicy` whose action head matches `BCPolicy.forward` (tanh+sigmoid) for pure 1:1 weight transfer, a PPO training CLI with Drive-backed resume, a PPO→ONNX export path extending `export_onnx.py`, and a go/no-go evaluation gate.

**Tech Stack:** Python 3.10+, `metadrive-simulator`, `stable-baselines3[extra]`, `gymnasium`, `torch`, `onnx`, `onnxruntime`, `opencv-python`, `numpy`. Existing `shared.preprocess`, `shared.dataset_format`, `training.imitation.model.BCPolicy`, and `training.imitation.train_bc` are reused unchanged.

**Testing approach:** All tests run under `pytest` from the `training/` venv. Tests that require `metadrive` use `pytest.importorskip("metadrive")` so the core test suite stays green on hosts without the heavy simulator installed. SB3/gymnasium tests similarly skip when unavailable.

---

## Task 0: Dependency scaffolding

**Files:**
- Create: `training/requirements-rl.txt`
- Create: `training/metadrive/__init__.py`
- Create: `training/metadrive/README.md`
- Modify: `training/README.md` (append a short section pointing at the new directory)

- [ ] **Step 1: Create `training/requirements-rl.txt`**

```
# Heavy deps for the MetaDrive RL pipeline. Kept separate from requirements.txt
# so the core training env (imitation + ONNX export) stays lightweight.
#
# Install:   pip install -r training/requirements-rl.txt
# (You still need training/requirements.txt for the shared bits.)

metadrive-simulator>=0.4.2.3
stable-baselines3[extra]>=2.3.0
gymnasium>=0.29
```

- [ ] **Step 2: Create `training/metadrive/__init__.py`**

```python
"""MetaDrive-based RL training for CouchMo.

See docs/superpowers/specs/2026-04-17-metadrive-rl-training-design.md for the
full design. Public modules:

- env           -- CouchMoMetaDriveEnv, a gym.Env wrapping SafeMetaDriveEnv
- expert_policy -- thin adapter around MetaDrive's IDM expert
- collect_bc    -- CLI that records expert rollouts as shared.dataset_format shards
- policy        -- CouchMoFeaturesExtractor + CouchMoActorCriticPolicy
- train_ppo     -- PPO fine-tune CLI with Drive-backed resume
- eval_policy   -- post-training go/no-go evaluation
"""
```

- [ ] **Step 3: Create `training/metadrive/README.md`**

```markdown
# CouchMo — MetaDrive RL training

This directory implements the MetaDrive-based RL pipeline specified in
`docs/superpowers/specs/2026-04-17-metadrive-rl-training-design.md`.

## Install (local dev)

```bash
cd training
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt -r requirements-rl.txt
```

## Pipeline

1. `python -m training.metadrive.collect_bc --data-root <path> --episodes 500`
2. `python -m training.imitation.train_bc --data-root <path> --out <bc.pt>`
3. `python -m training.metadrive.train_ppo --bc-ckpt <bc.pt> --output-dir <run_dir>`
4. `python -m training.imitation.export_onnx --from-ppo <run_dir/best.zip> --out <model.onnx>`
5. `python -m training.metadrive.eval_policy --onnx <model.onnx> --episodes 500`

See `colab_runner.ipynb` for the Colab-driven flow.
```

- [ ] **Step 4: Append a section to `training/README.md`**

At the end of `training/README.md` add:

```markdown

## MetaDrive RL pipeline

The `training/metadrive/` subpackage adds a sibling RL pipeline (BC pretrain +
PPO fine-tune in MetaDrive). It reuses this package's `BCPolicy`, `train_bc.py`,
and `export_onnx.py` as-is. Heavy deps live in `requirements-rl.txt`:

```bash
pip install -r requirements-rl.txt
```

See `training/metadrive/README.md` for the full flow.
```

- [ ] **Step 5: Commit**

```bash
git add training/requirements-rl.txt training/metadrive/__init__.py training/metadrive/README.md training/README.md
git commit -m "metadrive: scaffold training/metadrive/ package + requirements-rl.txt"
```

---

## Task 1: `CouchMoMetaDriveEnv` — minimum viable env (obs + action, no reward yet)

**Files:**
- Create: `training/metadrive/env.py`
- Create: `training/tests/test_metadrive_env.py`

This task gets the env wrapper producing correct observations and accepting correct actions. Reward shaping, termination, and domain randomization land in later tasks.

- [ ] **Step 1: Write the failing test**

Create `training/tests/test_metadrive_env.py`:

```python
"""Smoke tests for CouchMoMetaDriveEnv.

All tests skip if `metadrive` is not installed — this is the heavy optional
dep from `training/requirements-rl.txt`.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("metadrive")
pytest.importorskip("gymnasium")


def test_env_reset_returns_correct_obs_shape():
    from training.metadrive.env import CouchMoMetaDriveEnv

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 5})
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (8, 84, 84), f"expected (8,84,84); got {obs.shape}"
        assert obs.dtype == np.float32
        assert 0.0 <= obs.min() and obs.max() <= 1.0
        assert isinstance(info, dict)
    finally:
        env.close()


def test_env_step_returns_obs_reward_done_info():
    from training.metadrive.env import CouchMoMetaDriveEnv

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 5})
    try:
        env.reset(seed=0)
        action = np.array([0.0, 0.3], dtype=np.float32)  # steer=0, throttle=0.3
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (8, 84, 84)
        assert obs.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_env_action_space_matches_serial_protocol():
    from training.metadrive.env import CouchMoMetaDriveEnv

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 5})
    try:
        low, high = env.action_space.low, env.action_space.high
        assert np.allclose(low, [-1.0, 0.0]), f"expected [-1,0]; got {low}"
        assert np.allclose(high, [1.0, 1.0]), f"expected [1,1]; got {high}"
    finally:
        env.close()


def test_env_observation_space_shape():
    from training.metadrive.env import CouchMoMetaDriveEnv

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 5})
    try:
        assert env.observation_space.shape == (8, 84, 84)
        assert env.observation_space.dtype == np.float32
    finally:
        env.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd training && source .venv/bin/activate && python -m pytest tests/test_metadrive_env.py -v
```

Expected (on a host with `metadrive` installed): FAIL with `ModuleNotFoundError` or `ImportError: cannot import name 'CouchMoMetaDriveEnv'`. On a host without metadrive: tests are SKIPPED (also acceptable).

- [ ] **Step 3: Implement `CouchMoMetaDriveEnv`**

Create `training/metadrive/env.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd training && source .venv/bin/activate && python -m pytest tests/test_metadrive_env.py -v
```

Expected (with metadrive installed): 4 passed. Without metadrive: 4 skipped.

- [ ] **Step 5: Commit**

```bash
git add training/metadrive/env.py training/tests/test_metadrive_env.py
git commit -m "metadrive: CouchMoMetaDriveEnv obs + action wiring"
```

---

## Task 2: Reward shaping + termination

**Files:**
- Modify: `training/metadrive/env.py`
- Modify: `training/tests/test_metadrive_env.py`

Layers reward + termination onto the env from Task 1. Rewards come from MetaDrive's per-step signals plus smoothness/idle penalties computed from policy action history.

- [ ] **Step 1: Add failing tests**

Append to `training/tests/test_metadrive_env.py`:

```python
def test_reward_is_finite_on_normal_step():
    from training.metadrive.env import CouchMoMetaDriveEnv

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 5})
    try:
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(np.array([0.0, 0.3], dtype=np.float32))
        assert np.isfinite(reward)
    finally:
        env.close()


def test_reward_penalizes_idle_when_stopped():
    """Idle penalty should fire when velocity stays near zero."""
    from training.metadrive.env import CouchMoMetaDriveEnv

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 5})
    try:
        env.reset(seed=0)
        # Zero throttle -> vehicle should stay near zero velocity.
        rewards = []
        for _ in range(5):
            _, r, term, trunc, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
            rewards.append(r)
            if term or trunc:
                break
        # At least one reward should include a negative idle penalty.
        assert min(rewards) < 0, f"expected an idle-penalty step; got {rewards}"
    finally:
        env.close()


def test_truncation_at_max_steps():
    from training.metadrive.env import CouchMoMetaDriveEnv

    env = CouchMoMetaDriveEnv(
        config={"num_scenarios": 5, "horizon": 5}
    )
    try:
        env.reset(seed=0)
        truncated = False
        for _ in range(20):
            _, _, term, trunc, _ = env.step(np.array([0.0, 0.3], dtype=np.float32))
            if trunc:
                truncated = True
                break
            if term:
                break
        assert truncated or term, "expected termination or truncation within 20 steps"
    finally:
        env.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_metadrive_env.py::test_reward_is_finite_on_normal_step tests/test_metadrive_env.py::test_reward_penalizes_idle_when_stopped -v
```

Expected: at least the idle-penalty test FAILs (current env returns raw MetaDrive reward; no idle penalty).

- [ ] **Step 3: Add reward shaping + termination to `env.py`**

In `training/metadrive/env.py`, add these module-level constants under the existing constants:

```python
# Reward weights (see spec §Reward).
W_PROGRESS: float = 1.0
W_COLLISION: float = 50.0
W_OFF_ROAD: float = 20.0
W_SMOOTH: float = 0.1
W_IDLE: float = 0.05
IDLE_VEL_THRESHOLD: float = 0.1     # m/s
MAX_EPISODE_STEPS: int = 500
```

Then modify `CouchMoMetaDriveEnv.__init__` to initialize bookkeeping. After the `self._stacker = ...` line, add:

```python
        self._prev_action: np.ndarray = np.zeros(2, dtype=np.float32)
        self._prev_pos: np.ndarray | None = None
        self._step_count: int = 0
```

Modify `reset` to reset this state — replace the existing `reset` body with:

```python
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        md_obs, md_info = self._md_env.reset(seed=seed)
        self._stacker.reset()
        self._attach_cameras_to_ego()
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_pos = self._current_xy()
        self._step_count = 0
        obs = self._build_observation()
        return obs, dict(md_info)
```

Replace the existing `step` body with:

```python
    def step(self, action: np.ndarray):
        md_action = self._to_metadrive_action(action)
        _, _md_reward, md_terminated, md_truncated, info = self._md_env.step(md_action)

        self._step_count += 1
        reward, terminated = self._compute_reward_and_termination(action, info)
        truncated = md_truncated or self._step_count >= MAX_EPISODE_STEPS

        self._prev_action = action.astype(np.float32, copy=True)
        self._prev_pos = self._current_xy()

        obs = self._build_observation()
        return obs, float(reward), bool(terminated or md_terminated), bool(truncated), dict(info)
```

Add these helper methods to the class:

```python
    def _current_xy(self) -> np.ndarray:
        vehicle = self._md_env.engine.agent_manager.active_agents["default_agent"]
        pos = vehicle.position  # (x, y)
        return np.array([pos[0], pos[1]], dtype=np.float32)

    def _current_speed(self) -> float:
        vehicle = self._md_env.engine.agent_manager.active_agents["default_agent"]
        return float(vehicle.speed)  # m/s

    def _is_collision(self, info: dict) -> bool:
        vehicle = self._md_env.engine.agent_manager.active_agents["default_agent"]
        return bool(
            getattr(vehicle, "crash_vehicle", False)
            or getattr(vehicle, "crash_object", False)
            or getattr(vehicle, "crash_sidewalk", False)
        )

    def _is_off_road(self, info: dict) -> bool:
        vehicle = self._md_env.engine.agent_manager.active_agents["default_agent"]
        return bool(getattr(vehicle, "out_of_road", False))

    def _compute_reward_and_termination(
        self, action: np.ndarray, info: dict
    ) -> tuple[float, bool]:
        reward = 0.0

        # Progress along road (approximated by forward XY distance delta).
        cur_pos = self._current_xy()
        if self._prev_pos is not None:
            progress = float(np.linalg.norm(cur_pos - self._prev_pos))
            reward += W_PROGRESS * progress

        # Action smoothness (quadratic penalty on action delta).
        delta = action.astype(np.float32) - self._prev_action
        reward -= W_SMOOTH * float(np.dot(delta, delta))

        # Idle penalty (stopped but commanded to move? still penalize).
        if self._current_speed() < IDLE_VEL_THRESHOLD:
            reward -= W_IDLE

        # Terminal rewards.
        terminated = False
        if self._is_collision(info):
            reward -= W_COLLISION
            terminated = True
        elif self._is_off_road(info):
            reward -= W_OFF_ROAD
            terminated = True

        return reward, terminated
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_metadrive_env.py -v
```

Expected: all tests pass (or skip if metadrive missing).

- [ ] **Step 5: Commit**

```bash
git add training/metadrive/env.py training/tests/test_metadrive_env.py
git commit -m "metadrive: reward shaping + termination in CouchMoMetaDriveEnv"
```

---

## Task 3: Domain randomization (per-reset + per-step)

**Files:**
- Modify: `training/metadrive/env.py`
- Modify: `training/tests/test_metadrive_env.py`

Adds the randomization knobs from the spec §Domain randomization: reset-time camera/ego/steering-gain/brightness jitter, and per-step camera noise + action delay.

- [ ] **Step 1: Add failing tests**

Append to `training/tests/test_metadrive_env.py`:

```python
def test_two_resets_produce_different_camera_jitter():
    """Reset jitter (camera pose, steering gain) should diverge across seeds."""
    from training.metadrive.env import CouchMoMetaDriveEnv

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 10, "randomize": True})
    try:
        env.reset(seed=0)
        gain_0 = env._steering_gain
        env.reset(seed=1)
        gain_1 = env._steering_gain
        assert gain_0 != gain_1, "steering gain should differ across resets"
    finally:
        env.close()


def test_brightness_scale_bounded():
    from training.metadrive.env import CouchMoMetaDriveEnv

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 10, "randomize": True})
    try:
        for seed in range(5):
            env.reset(seed=seed)
            assert 0.7 <= env._brightness_scale <= 1.3
    finally:
        env.close()


def test_randomization_off_by_default():
    """Without the 'randomize' flag, steering gain is 1.0 and brightness is 1.0."""
    from training.metadrive.env import CouchMoMetaDriveEnv

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 5})
    try:
        env.reset(seed=0)
        assert env._steering_gain == 1.0
        assert env._brightness_scale == 1.0
    finally:
        env.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_metadrive_env.py::test_randomization_off_by_default -v
```

Expected: FAIL with `AttributeError: 'CouchMoMetaDriveEnv' object has no attribute '_steering_gain'`.

- [ ] **Step 3: Add randomization to `env.py`**

Add these constants near the other module-level constants:

```python
# Domain randomization ranges (see spec §Domain randomization).
STEER_GAIN_RANGE: tuple[float, float] = (0.85, 1.15)
BRIGHTNESS_RANGE: tuple[float, float] = (0.7, 1.3)
CAM_PITCH_JITTER_DEG: float = 3.0
CAM_LATERAL_JITTER_M: float = 0.02
EGO_LATERAL_OFFSET_M: float = 0.3
EGO_HEADING_OFFSET_DEG: float = 5.0
CAM_NOISE_STD: float = 3.0          # uint8 pixel units
ACTION_DELAY_PROB: float = 0.1
```

Modify `__init__` — after the existing assignments, add:

```python
        self._randomize: bool = bool(cfg.get("randomize", False))
        self._rng = np.random.default_rng(0)

        # Randomization state — reset per episode.
        self._steering_gain: float = 1.0
        self._brightness_scale: float = 1.0
        self._cam_pitch_offsets = (0.0, 0.0)   # (left_deg, right_deg)
        self._cam_lateral_offsets = (0.0, 0.0) # (left_m, right_m)
```

Modify `reset` — replace its body with:

```python
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
```

Add these helpers:

```python
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
```

Update `_attach_cameras_to_ego` to apply pitch + lateral offsets (replace the existing body):

```python
    def _attach_cameras_to_ego(self) -> None:
        engine = self._md_env.engine
        vehicle = engine.agent_manager.active_agents["default_agent"]
        left_cam = engine.get_sensor("left_cam")
        right_cam = engine.get_sensor("right_cam")

        left_pitch = CAM_PITCH_DEG + self._cam_pitch_offsets[0]
        right_pitch = CAM_PITCH_DEG + self._cam_pitch_offsets[1]
        left_lat = -CAM_LATERAL_M + self._cam_lateral_offsets[0]
        right_lat = +CAM_LATERAL_M + self._cam_lateral_offsets[1]

        left_cam.get_cam().setPos(vehicle.origin, left_lat, 0.3, CAM_HEIGHT_M)
        right_cam.get_cam().setPos(vehicle.origin, right_lat, 0.3, CAM_HEIGHT_M)
        left_cam.get_cam().setHpr(0, left_pitch, 0)
        right_cam.get_cam().setHpr(0, right_pitch, 0)
```

Update `_build_observation` to apply visual randomization:

```python
    def _build_observation(self) -> np.ndarray:
        left_bgr = self._apply_visual_randomization(self._read_camera_rgb("left_cam"))
        right_bgr = self._apply_visual_randomization(self._read_camera_rgb("right_cam"))
        pair = preprocess_pair(left_bgr, right_bgr)
        return self._stacker.push(pair)
```

Update `_to_metadrive_action` to apply steering gain + action delay:

```python
    def _to_metadrive_action(self, action: np.ndarray) -> np.ndarray:
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))

        if self._randomize:
            # Steering gain (bridges skid-steer dynamics gap).
            steer *= self._steering_gain
            steer = float(np.clip(steer, -1.0, 1.0))

            # Action delay — occasionally re-apply previous action.
            if self._rng.random() < ACTION_DELAY_PROB:
                steer = float(self._prev_action[0]) * self._steering_gain
                throttle = float(self._prev_action[1])

        return np.array([steer, throttle], dtype=np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_metadrive_env.py -v
```

Expected: all pass (or skip).

- [ ] **Step 5: Commit**

```bash
git add training/metadrive/env.py training/tests/test_metadrive_env.py
git commit -m "metadrive: domain randomization (per-reset jitter + per-step noise/delay)"
```

---

## Task 4: IDM expert policy adapter

**Files:**
- Create: `training/metadrive/expert_policy.py`
- Create: `training/tests/test_metadrive_expert.py`

Wraps MetaDrive's built-in IDM expert so it produces actions in our serial-protocol format `(steer ∈ [-1,1], throttle ∈ [0,1])`.

- [ ] **Step 1: Write the failing test**

Create `training/tests/test_metadrive_expert.py`:

```python
"""Tests for the MetaDrive IDM expert adapter."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("metadrive")


def test_expert_action_bounds():
    from training.metadrive.env import CouchMoMetaDriveEnv
    from training.metadrive.expert_policy import IDMExpertAdapter

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 5})
    try:
        env.reset(seed=0)
        expert = IDMExpertAdapter(env)
        for _ in range(5):
            action = expert.act()
            assert action.shape == (2,)
            assert -1.0 <= action[0] <= 1.0, f"steer out of bounds: {action[0]}"
            assert 0.0 <= action[1] <= 1.0, f"throttle out of bounds: {action[1]}"
            env.step(action)
    finally:
        env.close()


def test_expert_drives_forward_on_average():
    """The IDM expert should command positive throttle on most steps of a simple rollout."""
    from training.metadrive.env import CouchMoMetaDriveEnv
    from training.metadrive.expert_policy import IDMExpertAdapter

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 5})
    try:
        env.reset(seed=0)
        expert = IDMExpertAdapter(env)
        throttles = []
        for _ in range(20):
            action = expert.act()
            throttles.append(action[1])
            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        assert np.mean(throttles) > 0.05, f"mean throttle too low: {np.mean(throttles)}"
    finally:
        env.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_metadrive_expert.py -v
```

Expected: FAIL — module does not exist yet.

- [ ] **Step 3: Implement `expert_policy.py`**

Create `training/metadrive/expert_policy.py`:

```python
"""Adapter around MetaDrive's built-in IDM expert.

MetaDrive's `expert()` helper produces a 2D action [steering, throttle_brake]
both in [-1, 1], in the env's native action space. We clip throttle to [0, 1]
to match the CouchMo serial protocol — the IDM expert only rarely commands
negative throttle (brake) in the Safe scenarios; those few steps become
"coast" (throttle=0) in our dataset, which is fine.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    from metadrive.policy.idm_policy import IDMPolicy
except ImportError as exc:  # pragma: no cover — heavy optional dep
    raise ImportError(
        "metadrive is required; pip install -r training/requirements-rl.txt"
    ) from exc

if TYPE_CHECKING:
    from training.metadrive.env import CouchMoMetaDriveEnv


class IDMExpertAdapter:
    """Produces CouchMo-format (steer, throttle) from MetaDrive's IDM expert."""

    def __init__(self, env: "CouchMoMetaDriveEnv") -> None:
        self._env = env

    def act(self) -> np.ndarray:
        engine = self._env._md_env.engine
        vehicle = engine.agent_manager.active_agents["default_agent"]

        # IDMPolicy's action tensor is shape (2,): [steering, throttle_brake].
        policy = IDMPolicy(vehicle, random_seed=0)
        raw = policy.act(agent_id="default_agent")
        raw = np.asarray(raw, dtype=np.float32)

        steer = float(np.clip(raw[0], -1.0, 1.0))
        throttle = float(np.clip(raw[1], 0.0, 1.0))  # drop brake (negative)
        return np.array([steer, throttle], dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_metadrive_expert.py -v
```

Expected: 2 passed (or skipped).

- [ ] **Step 5: Commit**

```bash
git add training/metadrive/expert_policy.py training/tests/test_metadrive_expert.py
git commit -m "metadrive: IDM expert adapter -> (steer, throttle)"
```

---

## Task 5: BC data collection CLI

**Files:**
- Create: `training/metadrive/collect_bc.py`
- Create: `training/tests/test_metadrive_collect.py`

Rolls out the IDM expert for N episodes, logs raw BGR stereo frames + expert `(steer, throttle)` tuples, writes one shard per episode via `shared.dataset_format`, and updates a top-level `manifest.json`. The output directory is drop-in compatible with `training.imitation.train_bc.load_dataset`.

- [ ] **Step 1: Write the failing test**

Create `training/tests/test_metadrive_collect.py`:

```python
"""End-to-end smoke test for BC data collection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("metadrive")


def test_collect_bc_writes_valid_shards(tmp_path: Path):
    from training.metadrive.collect_bc import collect

    out = collect(
        data_root=tmp_path,
        episodes=2,
        max_steps=30,
        start_seed=0,
    )

    manifest_path = out / "manifest.json"
    assert manifest_path.exists(), "manifest.json not written"

    from shared.dataset_format import Manifest, read_shard

    manifest = Manifest.from_json(manifest_path)
    assert len(manifest.episodes) == 2, f"expected 2 episodes; got {len(manifest.episodes)}"

    # Read first shard and verify shape contract.
    ep = manifest.episodes[0]
    shard_dir = tmp_path / ep.shard_path
    shard_files = sorted(shard_dir.glob("shard_*.npz"))
    assert shard_files, f"no shards in {shard_dir}"

    shard = read_shard(shard_files[0])
    assert shard["left"].ndim == 4 and shard["left"].shape[-1] == 3
    assert shard["right"].ndim == 4 and shard["right"].shape[-1] == 3
    assert shard["left"].dtype == np.uint8
    assert shard["steer"].dtype == np.float32
    assert shard["throttle"].dtype == np.float32
    n = len(shard["left"])
    assert (
        len(shard["right"]) == n
        and len(shard["steer"]) == n
        and len(shard["throttle"]) == n
        and len(shard["t"]) == n
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_metadrive_collect.py -v
```

Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement `collect_bc.py`**

Create `training/metadrive/collect_bc.py`:

```python
"""Collect behavior cloning dataset by running MetaDrive's IDM expert.

CLI usage::

    python -m training.metadrive.collect_bc \
        --data-root ./training/data/metadrive_bc \
        --episodes 500 \
        --max-steps 500

Writes one shard per episode under ``<data-root>/episodes/<episode_id>/shard_0.npz``
plus a top-level ``manifest.json``. The output directory is drop-in compatible
with ``training.imitation.train_bc.load_dataset``.

Frames are stored as raw BGR uint8 (256x256x3 as produced by MetaDrive's
RGBCameras). Preprocessing + frame stacking is applied at training load time,
exactly as it is for simulation/-produced shards.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
import uuid
from pathlib import Path

import numpy as np

from shared.dataset_format import EpisodeMeta, Manifest, write_shard

log = logging.getLogger(__name__)


def collect(
    data_root: Path,
    episodes: int,
    max_steps: int,
    start_seed: int = 0,
) -> Path:
    """Roll out the IDM expert and write a BC dataset to ``data_root``.

    Returns the absolute data root path.
    """
    # Lazy imports — metadrive is a heavy optional dep.
    from training.metadrive.env import CouchMoMetaDriveEnv
    from training.metadrive.expert_policy import IDMExpertAdapter

    data_root = Path(data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    episodes_dir = data_root / "episodes"
    episodes_dir.mkdir(exist_ok=True)

    manifest = Manifest()

    env = CouchMoMetaDriveEnv(
        config={"num_scenarios": max(episodes, 100), "randomize": False}
    )
    try:
        for ep_idx in range(episodes):
            seed = start_seed + ep_idx
            env.reset(seed=seed)
            expert = IDMExpertAdapter(env)

            lefts: list[np.ndarray] = []
            rights: list[np.ndarray] = []
            steers: list[float] = []
            throttles: list[float] = []
            ts: list[float] = []

            for step in range(max_steps):
                # Read raw BGR for logging BEFORE preprocessing.
                left_bgr = env._read_camera_rgb("left_cam")
                right_bgr = env._read_camera_rgb("right_cam")

                action = expert.act()
                lefts.append(left_bgr)
                rights.append(right_bgr)
                steers.append(float(action[0]))
                throttles.append(float(action[1]))
                ts.append(step * 0.1)  # 10 Hz

                _, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break

            if not lefts:
                log.warning("Episode %d produced 0 samples; skipping.", ep_idx)
                continue

            ep_id = f"ep_{ep_idx:04d}_{uuid.uuid4().hex[:6]}"
            shard_rel = f"episodes/{ep_id}"
            shard_abs = data_root / shard_rel
            shard_abs.mkdir(parents=True, exist_ok=True)

            write_shard(
                shard_abs / "shard_0.npz",
                left=np.stack(lefts, axis=0),
                right=np.stack(rights, axis=0),
                steer=np.asarray(steers, dtype=np.float32),
                throttle=np.asarray(throttles, dtype=np.float32),
                t=np.asarray(ts, dtype=np.float32),
            )

            manifest.episodes.append(
                EpisodeMeta(
                    id=ep_id,
                    n_samples=len(lefts),
                    created_utc=_dt.datetime.utcnow().isoformat() + "Z",
                    shard_path=shard_rel,
                    world="metadrive_safe",
                    notes=f"seed={seed}",
                )
            )

            log.info("Episode %d/%d (seed=%d) -> %d samples", ep_idx + 1, episodes, seed, len(lefts))

        manifest.to_json(data_root / "manifest.json")
    finally:
        env.close()

    log.info("Wrote %d episodes to %s", len(manifest.episodes), data_root)
    return data_root


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect BC dataset from MetaDrive's IDM expert.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, required=True, help="Output directory.")
    p.add_argument("--episodes", type=int, default=500, help="Number of episodes.")
    p.add_argument("--max-steps", type=int, default=500, help="Max steps per episode.")
    p.add_argument("--start-seed", type=int, default=0, help="First episode seed.")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    args = _build_parser().parse_args(argv)
    try:
        out = collect(
            data_root=args.data_root,
            episodes=args.episodes,
            max_steps=args.max_steps,
            start_seed=args.start_seed,
        )
        print(f"BC dataset written -> {out}")
    except Exception as exc:  # noqa: BLE001 — top-level CLI catch
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_metadrive_collect.py -v
```

Expected: 1 passed (or skipped).

- [ ] **Step 5: Sanity check: BC loader consumes MetaDrive shards**

```bash
python -c "
from pathlib import Path
from training.metadrive.collect_bc import collect
from training.imitation.train_bc import load_dataset

out = collect(Path('/tmp/md_bc'), episodes=2, max_steps=20)
obs, targets = load_dataset(out)
print('OK', obs.shape, targets.shape)
"
```

Expected (with metadrive installed): `OK (N, 8, 84, 84) (N, 2)` where N is about 40 samples.

- [ ] **Step 6: Commit**

```bash
git add training/metadrive/collect_bc.py training/tests/test_metadrive_collect.py
git commit -m "metadrive: BC data collection CLI (IDM expert -> dataset shards)"
```

---

## Task 6: Custom SB3 policy with BC-compatible action head

**Files:**
- Create: `training/metadrive/policy.py`
- Create: `training/tests/test_metadrive_policy.py`

Defines `CouchMoFeaturesExtractor` (the conv+fc1 of `BCPolicy`) and `CouchMoActorCriticPolicy` (SB3 ActorCriticPolicy with a custom action head that applies `tanh` on steer and `sigmoid` on throttle — matching `BCPolicy.forward` exactly). Also provides `copy_bc_weights_into_policy()` for the 1:1 BC→PPO weight transfer.

- [ ] **Step 1: Write the failing test**

Create `training/tests/test_metadrive_policy.py`:

```python
"""Tests for the custom SB3 policy + BC weight transfer."""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")


def test_features_extractor_output_shape():
    from training.metadrive.policy import CouchMoFeaturesExtractor
    from gymnasium import spaces

    obs_space = spaces.Box(low=0.0, high=1.0, shape=(8, 84, 84), dtype=np.float32)
    extractor = CouchMoFeaturesExtractor(obs_space)
    x = torch.zeros(2, 8, 84, 84)
    feats = extractor(x)
    assert feats.shape == (2, 256), f"expected (2,256); got {feats.shape}"


def test_policy_forward_produces_bounded_action():
    from training.metadrive.policy import make_policy
    from gymnasium import spaces

    obs_space = spaces.Box(low=0.0, high=1.0, shape=(8, 84, 84), dtype=np.float32)
    act_space = spaces.Box(
        low=np.array([-1.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )
    policy = make_policy(obs_space, act_space, lr_schedule=lambda _: 3e-4)
    obs = torch.zeros(3, 8, 84, 84)
    with torch.no_grad():
        action, _value, _log_prob = policy(obs, deterministic=True)

    action_np = action.numpy()
    assert action_np.shape == (3, 2)
    assert (action_np[:, 0] >= -1.0).all() and (action_np[:, 0] <= 1.0).all()
    assert (action_np[:, 1] >= 0.0).all() and (action_np[:, 1] <= 1.0).all()


def test_bc_weight_transfer_matches_bc_forward():
    """After copy_bc_weights_into_policy, the policy's deterministic action must
    equal BCPolicy.forward to floating-point precision."""
    from gymnasium import spaces
    from training.imitation.model import BCPolicy
    from training.metadrive.policy import copy_bc_weights_into_policy, make_policy

    obs_space = spaces.Box(low=0.0, high=1.0, shape=(8, 84, 84), dtype=np.float32)
    act_space = spaces.Box(
        low=np.array([-1.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

    torch.manual_seed(0)
    bc = BCPolicy(in_channels=8).eval()
    policy = make_policy(obs_space, act_space, lr_schedule=lambda _: 3e-4)
    copy_bc_weights_into_policy(bc, policy)

    x = torch.rand(5, 8, 84, 84)
    with torch.no_grad():
        bc_out = bc(x).numpy()
        pol_action, _, _ = policy(x, deterministic=True)
        pol_out = pol_action.numpy()

    max_diff = float(np.max(np.abs(bc_out - pol_out)))
    assert max_diff < 1e-5, f"BC vs policy forward differs by {max_diff}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_metadrive_policy.py -v
```

Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement `policy.py`**

Create `training/metadrive/policy.py`:

```python
"""Custom SB3 policy whose action head matches BCPolicy.forward exactly.

Why custom: the standard SB3 Gaussian-with-tanh-squash output head applies the
same tanh to both action dims. We need tanh on steer and sigmoid on throttle
so that a BCPolicy checkpoint transfers into PPO as a pure 1:1 weight copy
(no rescaling, no sign mismatch). See spec §Phase 2.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from gymnasium import spaces
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.distributions import DiagGaussianDistribution
except ImportError as exc:  # pragma: no cover — heavy optional dep
    raise ImportError(
        "stable-baselines3 is required; pip install -r training/requirements-rl.txt"
    ) from exc

from training.imitation.model import BCPolicy

FEATURE_DIM: int = 256


class CouchMoFeaturesExtractor(BaseFeaturesExtractor):
    """Wraps BCPolicy.conv1/conv2/conv3/fc1 -> 256-dim feature vector."""

    def __init__(self, observation_space: spaces.Box) -> None:
        super().__init__(observation_space, features_dim=FEATURE_DIM)
        in_channels = int(observation_space.shape[0])
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            _dummy = torch.zeros(1, in_channels, 84, 84)
            _flat = self._conv_forward(_dummy).shape[1]

        self.fc1 = nn.Linear(_flat, FEATURE_DIM)

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._conv_forward(x)
        return F.relu(self.fc1(h))


class CouchMoActorCriticPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy with a BCPolicy-compatible action head.

    Action distribution: DiagGaussian with learnable log_std.
    Action head (determines the distribution mean):
        fc2: Linear(256, 2)
        steer_mean = tanh(fc2(h)[..., 0])
        throttle_mean = sigmoid(fc2(h)[..., 1])
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        lr_schedule: Callable[[float], float],
        **kwargs,
    ) -> None:
        kwargs.setdefault("features_extractor_class", CouchMoFeaturesExtractor)
        kwargs.setdefault("net_arch", [])  # no extra MLP; features_extractor goes straight to heads
        kwargs.setdefault("share_features_extractor", True)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        """Build value net + action head with BCPolicy-compatible activations."""
        # SB3's base _build would create a squashed Gaussian action head; we override.
        self.action_net = nn.Linear(FEATURE_DIM, 2)  # matches BCPolicy.fc2
        self.value_net = nn.Linear(FEATURE_DIM, 1)

        # Diagonal Gaussian with learnable log_std (one scalar per action dim).
        self.action_dist = DiagGaussianDistribution(action_dim=2)
        self.log_std = nn.Parameter(torch.zeros(2) - 0.5, requires_grad=True)

        # Optimizer covers all trainable params.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _action_mean(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.action_net(features)
        steer = torch.tanh(raw[:, 0:1])
        throttle = torch.sigmoid(raw[:, 1:2])
        return torch.cat([steer, throttle], dim=1)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        mean = self._action_mean(features)
        value = self.value_net(features)

        distribution = self.action_dist.proba_distribution(mean, self.log_std)
        if deterministic:
            action = mean
        else:
            action = distribution.sample()
        action = torch.clamp(
            action,
            torch.tensor(self.action_space.low, dtype=action.dtype, device=action.device),
            torch.tensor(self.action_space.high, dtype=action.dtype, device=action.device),
        )
        log_prob = distribution.log_prob(action)
        return action, value, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        mean = self._action_mean(features)
        value = self.value_net(features)
        distribution = self.action_dist.proba_distribution(mean, self.log_std)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return value, log_prob, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_net(self.extract_features(obs))


def make_policy(
    observation_space: spaces.Box,
    action_space: spaces.Box,
    lr_schedule: Callable[[float], float],
) -> CouchMoActorCriticPolicy:
    """Construct the custom policy. Useful from tests."""
    return CouchMoActorCriticPolicy(observation_space, action_space, lr_schedule)


def copy_bc_weights_into_policy(
    bc: BCPolicy, policy: CouchMoActorCriticPolicy
) -> list[str]:
    """Copy BCPolicy weights into the custom policy. Returns names of transferred tensors."""
    transferred: list[str] = []
    extractor = policy.features_extractor  # CouchMoFeaturesExtractor
    extractor.conv1.load_state_dict(bc.conv1.state_dict()); transferred.append("conv1")
    extractor.conv2.load_state_dict(bc.conv2.state_dict()); transferred.append("conv2")
    extractor.conv3.load_state_dict(bc.conv3.state_dict()); transferred.append("conv3")
    extractor.fc1.load_state_dict(bc.fc1.state_dict()); transferred.append("fc1")
    policy.action_net.load_state_dict(bc.fc2.state_dict()); transferred.append("action_net(<-fc2)")
    # value_net + log_std stay at their fresh init.
    return transferred
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_metadrive_policy.py -v
```

Expected: 3 passed (or skipped).

- [ ] **Step 5: Commit**

```bash
git add training/metadrive/policy.py training/tests/test_metadrive_policy.py
git commit -m "metadrive: CouchMoActorCriticPolicy with BC-compatible tanh/sigmoid action head"
```

---

## Task 7: PPO training CLI with Drive-backed resume + curriculum

**Files:**
- Create: `training/metadrive/train_ppo.py`
- Create: `training/tests/test_metadrive_train_ppo.py`

Builds the SB3 PPO trainer, wires BC weight transfer, checkpoint + eval callbacks, a curriculum callback that ramps traffic density and accident probability with training step count, and a resume path that scans the output directory for the latest checkpoint and continues from there.

- [ ] **Step 1: Write the failing test**

Create `training/tests/test_metadrive_train_ppo.py`:

```python
"""PPO training smoke test + resume roundtrip.

These tests run a tiny number of steps; they verify plumbing, not policy
quality. Still heavy — both `stable_baselines3` and `metadrive` are required.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("stable_baselines3")
pytest.importorskip("metadrive")


def test_train_ppo_writes_checkpoint_and_resumes(tmp_path: Path):
    from training.metadrive.train_ppo import train

    # Initial run — 128 steps, one checkpoint every 64 steps.
    first = train(
        bc_ckpt=None,
        output_dir=tmp_path / "run",
        total_timesteps=128,
        n_envs=1,
        checkpoint_freq=64,
        eval_freq=0,
        n_steps=32,
        batch_size=32,
    )
    assert first["num_timesteps"] >= 128

    ckpts = sorted((tmp_path / "run").glob("checkpoint_*.zip"))
    assert ckpts, "no checkpoints written"

    # Resume — request 256 total; should continue from 128 to 256, not restart.
    second = train(
        bc_ckpt=None,
        output_dir=tmp_path / "run",
        total_timesteps=256,
        n_envs=1,
        checkpoint_freq=64,
        eval_freq=0,
        n_steps=32,
        batch_size=32,
    )
    assert second["num_timesteps"] >= 256
    assert second["resumed_from"] is not None, "resume did not trigger"


def test_train_ppo_with_bc_init(tmp_path: Path):
    """Verify BC checkpoint init produces a policy that predicts finite actions."""
    from training.imitation.model import BCPolicy
    from training.metadrive.train_ppo import train

    # Write a dummy BC checkpoint.
    bc = BCPolicy(in_channels=8)
    bc_ckpt = tmp_path / "bc.pt"
    torch.save({"state_dict": bc.state_dict(), "arch": {"in_channels": 8}}, bc_ckpt)

    result = train(
        bc_ckpt=bc_ckpt,
        output_dir=tmp_path / "run",
        total_timesteps=64,
        n_envs=1,
        checkpoint_freq=64,
        eval_freq=0,
        n_steps=32,
        batch_size=32,
    )
    assert result["bc_transferred"], "BC transfer did not report transferred layers"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_metadrive_train_ppo.py -v
```

Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement `train_ppo.py`**

Create `training/metadrive/train_ppo.py`:

```python
"""PPO fine-tune on CouchMoMetaDriveEnv.

CLI usage::

    python -m training.metadrive.train_ppo \
        --bc-ckpt ./checkpoints/bc_metadrive.pt \
        --output-dir ./checkpoints/ppo_runs/run_001 \
        --total-timesteps 5000000

Features:
* Custom ActorCriticPolicy (BCPolicy-compatible action head).
* BC weight transfer via --bc-ckpt.
* SubprocVecEnv for parallel rollouts.
* CheckpointCallback + EvalCallback (held-out seeds).
* CurriculumCallback ramps traffic density + accident probability.
* Resume-from-checkpoint: scans --output-dir for the latest ``checkpoint_*.zip``
  and continues with ``reset_num_timesteps=False`` if found.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)

CHECKPOINT_PATTERN = re.compile(r"checkpoint_(\d+)_steps\.zip$")


def _latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for p in output_dir.glob("checkpoint_*.zip"):
        m = CHECKPOINT_PATTERN.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _make_env_factory(randomize: bool, scenario_start: int = 0):
    def _factory():
        from training.metadrive.env import CouchMoMetaDriveEnv

        return CouchMoMetaDriveEnv(
            config={
                "num_scenarios": 2000,
                "start_seed": scenario_start,
                "randomize": randomize,
            }
        )
    return _factory


def train(
    bc_ckpt: Path | None,
    output_dir: Path,
    total_timesteps: int,
    n_envs: int = 8,
    checkpoint_freq: int = 100_000,
    eval_freq: int = 50_000,
    eval_episodes: int = 20,
    n_steps: int = 256,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
) -> dict[str, Any]:
    """Run (or resume) a PPO fine-tune. Returns a small status dict for tests."""
    # Lazy imports — heavy optional deps.
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from training.imitation.model import BCPolicy
    from training.metadrive.policy import (
        CouchMoActorCriticPolicy,
        copy_bc_weights_into_policy,
    )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    train_env = vec_cls([_make_env_factory(randomize=True) for _ in range(n_envs)])

    resumed_from: Path | None = _latest_checkpoint(output_dir)

    if resumed_from is not None:
        log.info("Resuming from %s", resumed_from)
        model = PPO.load(str(resumed_from), env=train_env)
        bc_transferred: list[str] = []
    else:
        model = PPO(
            CouchMoActorCriticPolicy,
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=str(output_dir / "tb"),
            verbose=1,
        )
        bc_transferred = []
        if bc_ckpt is not None:
            ckpt = torch.load(str(bc_ckpt), weights_only=True)
            bc = BCPolicy(**ckpt["arch"])
            bc.load_state_dict(ckpt["state_dict"])
            bc_transferred = copy_bc_weights_into_policy(bc, model.policy)
            log.info("BC weights transferred: %s", bc_transferred)

    callbacks: list[BaseCallback] = []

    if checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, checkpoint_freq // max(1, n_envs)),
                save_path=str(output_dir),
                name_prefix="checkpoint",
            )
        )

    if eval_freq > 0:
        eval_env = DummyVecEnv([_make_env_factory(randomize=False, scenario_start=10_000)])
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(output_dir),
                eval_freq=max(1, eval_freq // max(1, n_envs)),
                n_eval_episodes=eval_episodes,
                deterministic=True,
                render=False,
                log_path=str(output_dir),
            )
        )

    callbacks.append(CurriculumCallback(train_env))

    remaining = total_timesteps - int(model.num_timesteps)
    if remaining > 0:
        model.learn(
            total_timesteps=remaining,
            callback=CallbackList(callbacks),
            reset_num_timesteps=False,
        )

    final = output_dir / "final.zip"
    model.save(str(final))

    train_env.close()

    return {
        "num_timesteps": int(model.num_timesteps),
        "resumed_from": str(resumed_from) if resumed_from else None,
        "bc_transferred": bc_transferred,
        "final_path": str(final),
    }


class CurriculumCallback:
    """Ramps env config between phases of training.

    Phases (spec §Domain randomization):
    * [0, 1M):       traffic_density=0.1, accident_prob=0.0
    * [1M, 3M):      traffic_density=0.3, accident_prob=0.2
    * [3M, inf):     traffic_density=0.4, accident_prob=0.3
    """

    def __init__(self, vec_env):
        # SB3 CallbackList auto-detects callbacks via BaseCallback; subclass inline.
        from stable_baselines3.common.callbacks import BaseCallback

        class _Impl(BaseCallback):
            def __init__(self, vec_env) -> None:
                super().__init__()
                self._vec_env = vec_env
                self._phase = -1

            def _on_step(self) -> bool:
                step = int(self.num_timesteps)
                phase = 0 if step < 1_000_000 else (1 if step < 3_000_000 else 2)
                if phase == self._phase:
                    return True
                self._phase = phase
                density, accident = {
                    0: (0.1, 0.0),
                    1: (0.3, 0.2),
                    2: (0.4, 0.3),
                }[phase]
                self._vec_env.env_method(
                    "_update_scenario_config",
                    traffic_density=density,
                    accident_prob=accident,
                )
                return True

        self._impl = _Impl(vec_env)

    def __getattr__(self, name):
        return getattr(self._impl, name)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PPO fine-tune on CouchMoMetaDriveEnv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bc-ckpt", type=Path, default=None, help="BC .pt checkpoint for warm start.")
    p.add_argument("--output-dir", type=Path, required=True, help="Where checkpoints / TB logs go.")
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    args = _build_parser().parse_args(argv)
    try:
        result = train(
            bc_ckpt=args.bc_ckpt,
            output_dir=args.output_dir,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            checkpoint_freq=args.checkpoint_freq,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        print(result)
    except Exception as exc:  # noqa: BLE001 — top-level CLI catch
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add `_update_scenario_config` hook to `env.py`**

The curriculum callback calls `env_method("_update_scenario_config", ...)` on the vec env. Add this method to `CouchMoMetaDriveEnv` in `training/metadrive/env.py` (place it near `_sample_episode_randomization`):

```python
    def _update_scenario_config(self, traffic_density: float, accident_prob: float) -> None:
        """Called by the curriculum callback to adjust scenario difficulty mid-run.

        The new values take effect on the next env.reset() — SafeMetaDriveEnv
        consumes its config dict on reset.
        """
        self._md_env.config["traffic_density"] = float(traffic_density)
        self._md_env.config["accident_prob"] = float(accident_prob)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_metadrive_train_ppo.py tests/test_metadrive_env.py -v
```

Expected: all pass (or skipped).

- [ ] **Step 6: Commit**

```bash
git add training/metadrive/train_ppo.py training/metadrive/env.py training/tests/test_metadrive_train_ppo.py
git commit -m "metadrive: PPO training CLI with BC init, resume, curriculum"
```

---

## Task 8: ONNX export from PPO checkpoint

**Files:**
- Modify: `training/imitation/export_onnx.py`
- Create: `training/tests/test_onnx_export_from_ppo.py`

Extends the existing BC ONNX export with a `--from-ppo PATH` flag. Loads an SB3 `.zip`, extracts `features_extractor.conv1/conv2/conv3/fc1` + `action_net`, copies weights into a fresh `BCPolicy`, and exports through the existing export path. Verification: ONNX output within `1e-5` of the SB3 model's deterministic output.

- [ ] **Step 1: Write the failing test**

Create `training/tests/test_onnx_export_from_ppo.py`:

```python
"""Tests the --from-ppo export path."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")


def _make_dummy_ppo(path: Path):
    """Build and save a fresh SB3 PPO zip with CouchMoActorCriticPolicy."""
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from training.metadrive.policy import CouchMoActorCriticPolicy

    obs_space = spaces.Box(low=0.0, high=1.0, shape=(8, 84, 84), dtype=np.float32)
    act_space = spaces.Box(
        low=np.array([-1.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

    class _StubEnv:
        observation_space = obs_space
        action_space = act_space
        metadata = {"render_modes": []}

        def reset(self, **_):
            return np.zeros((8, 84, 84), dtype=np.float32), {}

        def step(self, _a):
            return np.zeros((8, 84, 84), dtype=np.float32), 0.0, True, False, {}

        def close(self):
            pass

        def render(self):
            return None

    env = DummyVecEnv([lambda: _StubEnv()])
    model = PPO(CouchMoActorCriticPolicy, env, n_steps=8, batch_size=8, verbose=0)
    model.save(str(path))


def test_from_ppo_export_matches_sb3_output(tmp_path: Path):
    from training.imitation.export_onnx import export

    ppo_zip = tmp_path / "ppo.zip"
    _make_dummy_ppo(ppo_zip)

    onnx_path = tmp_path / "out.onnx"
    export(
        pt_path=ppo_zip,
        onnx_path=onnx_path,
        opset=17,
        verify=True,
        from_ppo=True,
    )

    # Additional numeric check: ONNX output must match SB3 model.predict within 1e-5.
    import onnxruntime as ort
    from stable_baselines3 import PPO

    model = PPO.load(str(ppo_zip))
    x = np.random.rand(4, 8, 84, 84).astype(np.float32)
    sb3_action, _ = model.predict(x, deterministic=True)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_action = sess.run(None, {"input": x})[0]

    max_diff = float(np.max(np.abs(sb3_action - onnx_action)))
    assert max_diff < 1e-5, f"ONNX vs SB3 max abs diff = {max_diff}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_onnx_export_from_ppo.py -v
```

Expected: FAIL — `export()` does not accept `from_ppo`.

- [ ] **Step 3: Modify `export_onnx.py`**

Change the `export()` signature in `training/imitation/export_onnx.py` to accept `from_ppo: bool = False` and add the PPO-loading branch.

Modify the function signature (currently starts at line 28):

```python
def export(
    pt_path: Path,
    onnx_path: Path | None = None,
    opset: int = 17,
    verify: bool = False,
    from_ppo: bool = False,
) -> Path:
```

Add a docstring note under `verify`:

```
        from_ppo:  If *True*, interpret ``pt_path`` as a stable-baselines3 PPO
                   ``.zip`` produced by ``training.metadrive.train_ppo``. The
                   custom policy's features_extractor (conv1/conv2/conv3/fc1)
                   and action_net are copied into a fresh BCPolicy, which is
                   what gets exported. Skips the bc_meta sibling check.
```

Replace the checkpoint-loading block (currently the section that loads `.pt` with `torch.load` and builds `BCPolicy`) with this branching logic. Locate the block starting with the `# Optionally validate bc_meta sibling ...` comment through `model.eval()` and replace it with:

```python
    if from_ppo:
        model, in_channels = _load_model_from_ppo(pt_path)
    else:
        # Optionally validate bc_meta sibling as a sanity check.
        meta_path = pt_path.with_name(pt_path.stem + "_meta.json")
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            schema = meta.get("schema", "")
            if schema != BC_META_SCHEMA:
                raise ValueError(
                    f"Sibling meta file {meta_path} has unexpected schema "
                    f"'{schema}'; expected '{BC_META_SCHEMA}'. Is this a Task-8a checkpoint?"
                )

        ckpt = torch.load(str(pt_path), weights_only=True)

        if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "arch" not in ckpt:
            raise ValueError(
                "Checkpoint format not recognized; expected Task-8a format "
                "{'state_dict': ..., 'arch': {'in_channels': int}}. "
                f"Got type={type(ckpt).__name__}, keys={list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A'}"
            )

        arch = ckpt["arch"]
        model = BCPolicy(**arch)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        in_channels = arch.get("in_channels", _IN_CHANNELS)
```

Add the helper function `_load_model_from_ppo` near the top of the file (after the imports, before `export`):

```python
def _load_model_from_ppo(zip_path: Path) -> tuple[BCPolicy, int]:
    """Load an SB3 PPO zip and return (BCPolicy with PPO weights copied in, in_channels).

    Layer mapping (see training.metadrive.policy):
        features_extractor.conv1  -> BCPolicy.conv1
        features_extractor.conv2  -> BCPolicy.conv2
        features_extractor.conv3  -> BCPolicy.conv3
        features_extractor.fc1    -> BCPolicy.fc1
        action_net                -> BCPolicy.fc2
    """
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for --from-ppo; "
            "pip install -r training/requirements-rl.txt"
        ) from exc

    ppo = PPO.load(str(zip_path))
    extractor = ppo.policy.features_extractor
    in_channels = int(extractor.conv1.in_channels)

    model = BCPolicy(in_channels=in_channels)
    model.conv1.load_state_dict(extractor.conv1.state_dict())
    model.conv2.load_state_dict(extractor.conv2.state_dict())
    model.conv3.load_state_dict(extractor.conv3.state_dict())
    model.fc1.load_state_dict(extractor.fc1.state_dict())
    model.fc2.load_state_dict(ppo.policy.action_net.state_dict())
    model.eval()
    return model, in_channels
```

Extend the CLI parser (in `main`) with `--from-ppo`:

```python
    parser.add_argument(
        "--from-ppo",
        action="store_true",
        help="Interpret --in as an SB3 PPO .zip checkpoint instead of a Task-8a .pt.",
    )
```

And forward it to `export()` in the `main()` `try` block:

```python
        out = export(
            pt_path=args.pt_path,
            onnx_path=args.onnx_path,
            opset=args.opset,
            verify=args.verify,
            from_ppo=args.from_ppo,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_onnx_export_from_ppo.py tests/test_onnx_export.py -v
```

Expected: all pass (or skip when deps missing). The existing BC-export test stays green because we only added the branch and its default is `False`.

- [ ] **Step 5: Commit**

```bash
git add training/imitation/export_onnx.py training/tests/test_onnx_export_from_ppo.py
git commit -m "export_onnx: --from-ppo flag for SB3 PPO -> BCPolicy -> ONNX"
```

---

## Task 9: Post-training evaluation CLI (go/no-go gate)

**Files:**
- Create: `training/metadrive/eval_policy.py`
- Create: `training/tests/test_metadrive_eval.py`

Loads an ONNX artifact, rolls out N deterministic episodes with `randomize=False`, computes collision rate / off-road rate / mean episode length, and exits 0 if the thresholds from spec §R3 are met (1 otherwise). This is the go/no-go gate before real-hardware work.

- [ ] **Step 1: Write the failing test**

Create `training/tests/test_metadrive_eval.py`:

```python
"""Tests for the eval_policy CLI."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("metadrive")


def _make_dummy_onnx(path: Path, seed: int = 0) -> None:
    """Export a freshly-initialized BCPolicy to ONNX for use in smoke tests."""
    from training.imitation.export_onnx import export
    from training.imitation.model import BCPolicy

    torch.manual_seed(seed)
    bc = BCPolicy(in_channels=8)
    pt = path.with_suffix(".pt")
    torch.save({"state_dict": bc.state_dict(), "arch": {"in_channels": 8}}, pt)
    export(pt_path=pt, onnx_path=path, opset=17, verify=False)


def test_eval_policy_reports_metrics(tmp_path: Path):
    from training.metadrive.eval_policy import evaluate

    onnx = tmp_path / "model.onnx"
    _make_dummy_onnx(onnx)

    report = evaluate(onnx_path=onnx, episodes=2, max_steps=30, start_seed=10_000)
    for key in ("collision_rate", "off_road_rate", "mean_episode_length", "episodes"):
        assert key in report, f"missing key {key} in report: {report}"
    assert report["episodes"] == 2
    assert 0.0 <= report["collision_rate"] <= 1.0
    assert 0.0 <= report["off_road_rate"] <= 1.0
    assert report["mean_episode_length"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_metadrive_eval.py -v
```

Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement `eval_policy.py`**

Create `training/metadrive/eval_policy.py`:

```python
"""Evaluate an exported ONNX policy in MetaDrive and enforce go/no-go thresholds.

CLI::

    python -m training.metadrive.eval_policy \
        --onnx ./exports/couchmo_v1.onnx \
        --episodes 500

Exit codes:
    0 — all thresholds met (see GO_NO_GO below)
    1 — any threshold failed
    2 — runtime error

Thresholds from spec §R3.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Go/no-go thresholds (spec §R3).
GO_NO_GO = {
    "collision_rate_max": 0.05,
    "off_road_rate_max": 0.10,
    "mean_episode_length_min": 300.0,
}


def evaluate(
    onnx_path: Path,
    episodes: int,
    max_steps: int = 500,
    start_seed: int = 10_000,
) -> dict[str, Any]:
    """Run deterministic held-out eval and return a metrics dict."""
    import onnxruntime as ort
    from training.metadrive.env import CouchMoMetaDriveEnv

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 2000, "randomize": False})

    collisions = 0
    off_roads = 0
    lengths: list[int] = []

    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=start_seed + ep)
            ep_len = 0
            for step in range(max_steps):
                x = obs[np.newaxis, :, :, :].astype(np.float32)
                action = sess.run(None, {input_name: x})[0][0]
                obs, _, term, trunc, info = env.step(action)
                ep_len = step + 1
                if term or trunc:
                    if env._is_collision(info):
                        collisions += 1
                    elif env._is_off_road(info):
                        off_roads += 1
                    break
            lengths.append(ep_len)
            log.info("Episode %d/%d length=%d", ep + 1, episodes, ep_len)
    finally:
        env.close()

    return {
        "episodes": episodes,
        "collision_rate": collisions / episodes,
        "off_road_rate": off_roads / episodes,
        "mean_episode_length": float(np.mean(lengths)),
    }


def check_thresholds(report: dict[str, Any]) -> list[str]:
    """Return a list of human-readable failure strings (empty list = pass)."""
    failures: list[str] = []
    if report["collision_rate"] > GO_NO_GO["collision_rate_max"]:
        failures.append(
            f"collision_rate {report['collision_rate']:.3f} > "
            f"{GO_NO_GO['collision_rate_max']:.3f}"
        )
    if report["off_road_rate"] > GO_NO_GO["off_road_rate_max"]:
        failures.append(
            f"off_road_rate {report['off_road_rate']:.3f} > "
            f"{GO_NO_GO['off_road_rate_max']:.3f}"
        )
    if report["mean_episode_length"] < GO_NO_GO["mean_episode_length_min"]:
        failures.append(
            f"mean_episode_length {report['mean_episode_length']:.1f} < "
            f"{GO_NO_GO['mean_episode_length_min']:.1f}"
        )
    return failures


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate an exported ONNX policy in MetaDrive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--onnx", type=Path, required=True, help="Path to the exported ONNX.")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--start-seed", type=int, default=10_000, help="First eval seed (held out from training).")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    args = _build_parser().parse_args(argv)

    try:
        report = evaluate(
            onnx_path=args.onnx,
            episodes=args.episodes,
            max_steps=args.max_steps,
            start_seed=args.start_seed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    print(json.dumps(report, indent=2))

    failures = check_thresholds(report)
    if failures:
        print("GO/NO-GO FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    print("GO/NO-GO PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_metadrive_eval.py -v
```

Expected: 1 passed (or skipped).

- [ ] **Step 5: Commit**

```bash
git add training/metadrive/eval_policy.py training/tests/test_metadrive_eval.py
git commit -m "metadrive: eval_policy CLI (ONNX rollout + go/no-go gate)"
```

---

## Task 10: Colab runner notebook

**Files:**
- Create: `training/metadrive/colab_runner.ipynb`

A thin orchestration notebook with the 12 cells described in spec §Colab integration. No logic lives in the notebook — every cell is either a shell command, a module invocation, or a `%load_ext tensorboard`.

- [ ] **Step 1: Write the notebook**

Create `training/metadrive/colab_runner.ipynb` with the following JSON content (paste as a single file):

```json
{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# CouchMo — MetaDrive RL training (Colab)\n", "\n", "Thin orchestration notebook. See `training/metadrive/README.md` for context."]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["from google.colab import drive\n", "drive.mount('/content/drive')"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["%%bash\n", "cd /content\n", "if [ ! -d CouchMo ]; then\n", "  git clone https://github.com/<USER>/CouchMo.git\n", "else\n", "  cd CouchMo && git pull\n", "fi"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["%cd /content/CouchMo"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["!pip install -q -r training/requirements.txt -r training/requirements-rl.txt"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["# Smoke test: env reset/step on a single scenario.\n", "!python -c \"from training.metadrive.env import CouchMoMetaDriveEnv; import numpy as np\\nenv = CouchMoMetaDriveEnv(config={'num_scenarios': 5}); obs, _ = env.reset(seed=0); print('obs', obs.shape, obs.dtype); env.step(np.array([0.0, 0.3], dtype=np.float32)); env.close(); print('OK')\""]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["# BC data collection — skip if Drive already has the dataset.\n", "import pathlib\n", "DATA = '/content/drive/MyDrive/CouchMo/metadrive_bc'\n", "if not (pathlib.Path(DATA) / 'manifest.json').exists():\n", "    !python -m training.metadrive.collect_bc --data-root {DATA} --episodes 500 --max-steps 500\n", "else:\n", "    print('BC dataset already on Drive at', DATA)"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["# BC training.\n", "!python -m training.imitation.train_bc --data-root /content/drive/MyDrive/CouchMo/metadrive_bc --out /content/drive/MyDrive/CouchMo/checkpoints/bc_metadrive.pt --epochs 10"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["# PPO fine-tune. Auto-resumes if checkpoints exist under --output-dir.\n", "!python -m training.metadrive.train_ppo --bc-ckpt /content/drive/MyDrive/CouchMo/checkpoints/bc_metadrive.pt --output-dir /content/drive/MyDrive/CouchMo/checkpoints/ppo_runs/run_001 --total-timesteps 5000000 --n-envs 8"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["%load_ext tensorboard\n", "%tensorboard --logdir /content/drive/MyDrive/CouchMo/checkpoints/ppo_runs/run_001/tb"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["# Export best checkpoint to ONNX.\n", "!python -m training.imitation.export_onnx --from-ppo --in /content/drive/MyDrive/CouchMo/checkpoints/ppo_runs/run_001/best_model.zip --out /content/drive/MyDrive/CouchMo/exports/couchmo_v1.onnx --verify"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["# Go/no-go gate. Non-zero exit = not ready for real hardware.\n", "!python -m training.metadrive.eval_policy --onnx /content/drive/MyDrive/CouchMo/exports/couchmo_v1.onnx --episodes 500"]}
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Verify the notebook is valid JSON**

```bash
python -c "import json; json.loads(open('training/metadrive/colab_runner.ipynb').read()); print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add training/metadrive/colab_runner.ipynb
git commit -m "metadrive: Colab runner notebook (thin orchestration only)"
```

---

## Task 11: Integration smoke test + docs polish

**Files:**
- Create: `training/tests/test_metadrive_integration.py`
- Modify: `training/metadrive/README.md`

End-to-end smoke test that runs the whole pipeline at minimum scale: collect 2 episodes, train BC for 1 epoch on them, train PPO for 64 steps init-from-BC, export ONNX, evaluate for 2 episodes. This catches cross-module breakages that the per-module tests miss.

- [ ] **Step 1: Write the failing test**

Create `training/tests/test_metadrive_integration.py`:

```python
"""End-to-end smoke test for the MetaDrive pipeline."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("metadrive")
pytest.importorskip("stable_baselines3")


def test_full_pipeline_smoke(tmp_path: Path):
    from training.imitation.export_onnx import export
    from training.imitation.train_bc import main as train_bc_main
    from training.metadrive.collect_bc import collect
    from training.metadrive.eval_policy import evaluate
    from training.metadrive.train_ppo import train as train_ppo

    data_root = tmp_path / "bc_data"
    bc_ckpt = tmp_path / "bc.pt"
    ppo_dir = tmp_path / "ppo"
    onnx_path = tmp_path / "model.onnx"

    # 1. Collect tiny BC dataset.
    collect(data_root=data_root, episodes=2, max_steps=30)

    # 2. Train BC for 1 epoch.
    train_bc_main([
        "--data-root", str(data_root),
        "--epochs", "1",
        "--batch-size", "8",
        "--out", str(bc_ckpt),
        "--val-split", "0.2",
    ])
    assert bc_ckpt.exists()

    # 3. PPO fine-tune (64 steps, 1 env).
    result = train_ppo(
        bc_ckpt=bc_ckpt,
        output_dir=ppo_dir,
        total_timesteps=64,
        n_envs=1,
        checkpoint_freq=64,
        eval_freq=0,
        n_steps=32,
        batch_size=32,
    )
    assert result["bc_transferred"], "BC transfer did not fire"

    # 4. Export final PPO checkpoint to ONNX.
    export(
        pt_path=Path(result["final_path"]),
        onnx_path=onnx_path,
        opset=17,
        verify=True,
        from_ppo=True,
    )
    assert onnx_path.exists()

    # 5. Evaluate for 2 episodes (smoke).
    report = evaluate(onnx_path=onnx_path, episodes=2, max_steps=30)
    assert report["episodes"] == 2
```

- [ ] **Step 2: Run test to verify it passes**

```bash
python -m pytest tests/test_metadrive_integration.py -v
```

Expected: 1 passed (or skipped). On a laptop without GPU this may take 5–10 minutes.

- [ ] **Step 3: Flesh out `training/metadrive/README.md`**

Replace the minimal README from Task 0 with a fuller version:

```markdown
# CouchMo — MetaDrive RL training

Implements the pipeline specified in
`docs/superpowers/specs/2026-04-17-metadrive-rl-training-design.md`:

```
IDM expert rollouts  ->  BC pretrain  ->  PPO fine-tune  ->  ONNX  ->  serial_controller.py
```

## Install (local dev)

```bash
cd training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt -r requirements-rl.txt
```

## Local smoke test

```bash
python -m pytest tests/test_metadrive_integration.py -v
```

## Colab flow

Open `colab_runner.ipynb` in Colab and run cells top to bottom.

## Module-by-module

| Module | Purpose |
|---|---|
| `env.py` | Gym env wrapping `SafeMetaDriveEnv`; stereo cam obs + `(steer, throttle)` action. |
| `expert_policy.py` | IDM expert adapter; produces actions in serial-protocol format. |
| `collect_bc.py` | CLI: roll out expert, write shards via `shared.dataset_format`. |
| `policy.py` | `CouchMoFeaturesExtractor` + `CouchMoActorCriticPolicy` (tanh+sigmoid action head for 1:1 BC weight transfer). |
| `train_ppo.py` | SB3 PPO CLI with BC warm start, curriculum, resume from Drive. |
| `eval_policy.py` | ONNX evaluation with R3 go/no-go thresholds; non-zero exit if thresholds fail. |
| `colab_runner.ipynb` | Thin Colab orchestration. |

## Go/no-go gate

Before any real-hardware work, `eval_policy.py` must exit 0 on ≥500 held-out
episodes. Thresholds live in `eval_policy.GO_NO_GO`.

## Real-hardware rollout

See spec §Real-hardware rollout. Stages R0 (replay mode), R1 (bench),
R2 (parking lot, capped), R3 (cones), R4 (sidewalk pilot).
```

- [ ] **Step 4: Commit**

```bash
git add training/tests/test_metadrive_integration.py training/metadrive/README.md
git commit -m "metadrive: end-to-end integration smoke test + README"
```

---

## Self-review

**1. Spec coverage:**
- R1 (obs/action contract at 10 Hz) — Task 1 (obs space + action space), env uses `decision_repeat=10`.
- R2 (checkpoint resumability) — Task 7 `_latest_checkpoint` + `reset_num_timesteps=False`.
- R3 (go/no-go thresholds) — Task 9 `GO_NO_GO` + `check_thresholds`.
- Domain randomization — Task 3.
- Curriculum — Task 7 `CurriculumCallback`.
- BC-compatible action head (1:1 weight transfer) — Task 6 (`CouchMoActorCriticPolicy`) + test asserts `max_diff < 1e-5`.
- ONNX export preserves contract — Task 8 test asserts SB3 vs ONNX match within 1e-5.
- Colab orchestration — Task 10.
- Reuse existing `BCPolicy` / `train_bc.py` / `shared.preprocess` / `shared.dataset_format` — verified in Tasks 5 and 11 (sanity check runs `train_bc.load_dataset` on MetaDrive shards).
- Phased real-hardware rollout (R0–R4) — documented in README (Task 11); no code tasks needed since the limiter is a one-line edit in `runtime/inference.py` that should be made at R2 time, not now.

**2. Placeholder scan:**
- No `TBD`, `TODO`, "implement later", or generic "add error handling" steps. Every step contains the actual code or command to run.
- Task 10 uses `<USER>` in the `git clone` line — this is a legitimate placeholder for the user's GitHub handle and must be filled in when the notebook is actually run. Flagged as such.

**3. Type consistency:**
- `CouchMoMetaDriveEnv.__init__(config=...)` signature is consistent across tasks 1–3.
- `_is_collision(info)` / `_is_off_road(info)` helper names match between `env.py` (Task 2) and `eval_policy.py` (Task 9).
- `CouchMoFeaturesExtractor` has `conv1/conv2/conv3/fc1` — matches what `_load_model_from_ppo` expects in Task 8.
- `copy_bc_weights_into_policy(bc, policy)` signature matches call site in `train_ppo.py` Task 7.
- `train()` signature in Task 7 matches the call in Task 11 integration test.
- `evaluate()` signature in Task 9 matches the call in Task 11.
- `GO_NO_GO` dict keys (`collision_rate_max`, `off_road_rate_max`, `mean_episode_length_min`) match the metric keys used in `check_thresholds`.

**4. Scope:** one spec → one plan → one pipeline. No decomposition needed.
