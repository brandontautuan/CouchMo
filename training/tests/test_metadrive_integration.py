"""End-to-end smoke test for the MetaDrive pipeline."""
from __future__ import annotations

from pathlib import Path

import pytest

# importorskip must fire before any hard torch import so collection succeeds
# on torchless hosts. Matches Task 6/7/8/9 pattern.
torch = pytest.importorskip("torch")
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
