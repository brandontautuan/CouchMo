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
        # use_sde is silently overwritten by our _build override, so lock it off
        # to prevent a caller's gSDE request from being discarded without warning.
        if kwargs.get("use_sde", False):
            raise ValueError("CouchMoActorCriticPolicy does not support use_sde=True")
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        """Build value net + action head with BCPolicy-compatible activations."""
        # SB3's base _build would create a squashed Gaussian action head; we override.
        self.action_net = nn.Linear(FEATURE_DIM, 2)  # matches BCPolicy.fc2
        self.value_net = nn.Linear(FEATURE_DIM, 1)

        # Diagonal Gaussian with learnable log_std (one scalar per action dim).
        # log_std_init = -0.5 -> σ ≈ 0.61; a tighter prior than SB3's default 0.0
        # (σ = 1.0), chosen for the BC warm-start case where the mean is already
        # close to the expert and we want exploration to be modest.
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
