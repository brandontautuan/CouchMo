"""Tests for the MetaDrive IDM expert adapter."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("metadrive")
pytest.importorskip("gymnasium")


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
