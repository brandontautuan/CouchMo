"""Tests the --from-ppo export path."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Guard at import time — torch must be present. importorskip must precede any
# hard torch import, or pytest collection fails on torchless hosts.
torch = pytest.importorskip("torch")
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
