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
    """Produces CouchMo-format (steer, throttle) from MetaDrive's IDM expert.

    IDMPolicy is stateful (PID controllers, lane-routing memory), so we bind
    one instance to the current ego vehicle and reuse it across steps. When the
    underlying env is reset, MetaDrive may swap in a new vehicle object — we
    detect that by identity and rebuild the policy, destroying the old one so
    Panda3D tasks registered by the prior instance don't leak.
    """

    def __init__(self, env: "CouchMoMetaDriveEnv") -> None:
        self._env = env
        self._policy: IDMPolicy | None = None
        self._vehicle = None

    def act(self) -> np.ndarray:
        engine = self._env._md_env.engine
        vehicle = engine.agent_manager.active_agents["default_agent"]

        if self._vehicle is not vehicle:
            if self._policy is not None:
                destroy = getattr(self._policy, "destroy", None)
                if callable(destroy):
                    destroy()
            self._policy = IDMPolicy(vehicle, random_seed=0)
            self._vehicle = vehicle

        # IDMPolicy.act ignores *args/**kwargs and reads state from
        # self.control_object, so call it bare.
        raw = np.asarray(self._policy.act(), dtype=np.float32)

        steer = float(np.clip(raw[0], -1.0, 1.0))
        throttle = float(np.clip(raw[1], 0.0, 1.0))  # drop brake (negative)
        return np.array([steer, throttle], dtype=np.float32)
