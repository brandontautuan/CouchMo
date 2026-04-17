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
