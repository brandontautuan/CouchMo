"""PPO training smoke test + resume roundtrip.

These tests run a tiny number of steps; they verify plumbing, not policy
quality. Still heavy — both `stable_baselines3` and `metadrive` are required.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# importorskip must fire before `import torch` so this file collects cleanly
# on machines where torch (a heavy RL dep) is absent.
pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
pytest.importorskip("metadrive")

import torch  # noqa: E402 — intentional post-importorskip import


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
    """Smoke-test BC warm start. Deep weight-transfer equivalence is covered by
    test_bc_weight_transfer_matches_bc_forward in test_metadrive_policy.py;
    this test only verifies the train() wrapper wires the transfer through."""
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
