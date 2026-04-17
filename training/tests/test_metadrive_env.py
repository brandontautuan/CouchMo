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


def test_reward_is_finite_on_normal_step():
    from training.metadrive.env import CouchMoMetaDriveEnv

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 5})
    try:
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(np.array([0.0, 0.3], dtype=np.float32))
        assert np.isfinite(reward)
    finally:
        env.close()


def test_reward_penalizes_idle_when_stopped(monkeypatch):
    """Idle penalty path fires when _current_speed < IDLE_VEL_THRESHOLD.

    Isolates the idle branch by stubbing _current_speed to force the
    below-threshold path, so the assertion cannot pass for other reasons
    (progress, smoothness, or MetaDrive vehicle dynamics).
    """
    from training.metadrive.env import CouchMoMetaDriveEnv, W_IDLE

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 5})
    try:
        env.reset(seed=0)
        # Force the idle branch to fire deterministically.
        monkeypatch.setattr(env, "_current_speed", lambda: 0.0)
        # Zero action against zero _prev_action → no smoothness penalty.
        # _prev_pos is re-seeded to _current_xy() *after* reward calc, so
        # over two steps the progress term is at most one physics-step of
        # drift — typically zero when we command zero throttle.
        action = np.array([0.0, 0.0], dtype=np.float32)
        _, r, _, _, _ = env.step(action)
        # Reward for this step must include -W_IDLE. Progress and smoothness
        # contributions are non-negative and near-zero respectively for this
        # input, so r ≤ -W_IDLE + epsilon.
        assert r <= -W_IDLE + 1e-3, (
            f"expected reward ≤ -W_IDLE (~{-W_IDLE:.3f}); got {r:.4f}"
        )
    finally:
        env.close()


def test_truncation_at_max_episode_steps(monkeypatch):
    """MAX_EPISODE_STEPS branch fires independently of MetaDrive's horizon."""
    from training.metadrive import env as env_module
    from training.metadrive.env import CouchMoMetaDriveEnv

    # Shrink MAX_EPISODE_STEPS to 3 so we can hit it in seconds.
    # MetaDrive horizon is left at its default (much larger) so md_truncated
    # cannot fire first — we're guarding our own cap.
    monkeypatch.setattr(env_module, "MAX_EPISODE_STEPS", 3)

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 5})
    try:
        env.reset(seed=0)
        truncated_at = None
        for i in range(1, 10):
            _, _, term, trunc, _ = env.step(np.array([0.0, 0.3], dtype=np.float32))
            if trunc:
                truncated_at = i
                break
            if term:
                break
        assert truncated_at == 3, (
            f"expected truncation at step 3 (MAX_EPISODE_STEPS); got {truncated_at}"
        )
    finally:
        env.close()


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


def test_visual_randomization_respects_uint8_bounds():
    """With randomize=True, _apply_visual_randomization stays in [0,255] uint8
    and actually modifies the input (brightness scale + noise are non-trivial)."""
    from training.metadrive.env import CouchMoMetaDriveEnv

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 5, "randomize": True})
    try:
        env.reset(seed=0)
        fake = np.full((16, 16, 3), 128, dtype=np.uint8)
        out = env._apply_visual_randomization(fake)
        assert out.shape == fake.shape
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255
        assert not np.array_equal(out, fake), (
            "expected randomized output to differ from input"
        )
    finally:
        env.close()


def test_visual_randomization_is_identity_when_disabled():
    """Without randomize, _apply_visual_randomization returns the input unchanged."""
    from training.metadrive.env import CouchMoMetaDriveEnv

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 5})
    try:
        env.reset(seed=0)
        fake = np.full((16, 16, 3), 200, dtype=np.uint8)
        out = env._apply_visual_randomization(fake)
        assert np.array_equal(out, fake), "expected identity when randomize=False"
    finally:
        env.close()


def test_action_delay_branch_stays_clipped(monkeypatch):
    """Even when action-delay replays prev_action at max steer with max gain,
    the emitted MetaDrive action stays within [-1, 1]."""
    from training.metadrive.env import CouchMoMetaDriveEnv

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 5, "randomize": True})
    try:
        env.reset(seed=0)
        # Force worst case: gain = upper bound, prev steer = full lock.
        env._steering_gain = 1.15
        env._prev_action = np.array([1.0, 0.5], dtype=np.float32)
        # Force the action-delay branch to fire deterministically.
        monkeypatch.setattr(env._rng, "random", lambda: 0.0)
        out = env._to_metadrive_action(np.array([1.0, 0.5], dtype=np.float32))
        assert -1.0 <= out[0] <= 1.0, f"steer {out[0]} exceeded [-1,1]"
        assert 0.0 <= out[1] <= 1.0, f"throttle {out[1]} exceeded [0,1]"
    finally:
        env.close()
