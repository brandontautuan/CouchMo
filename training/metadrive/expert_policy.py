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
