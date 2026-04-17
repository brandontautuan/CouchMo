"""Tests for the custom SB3 policy + BC weight transfer."""
from __future__ import annotations

import numpy as np
import pytest

# importorskip must fire before `import torch` so this file collects cleanly
# on machines where torch (a heavy RL dep) is absent.
pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")

import torch  # noqa: E402 — intentional post-importorskip import


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
