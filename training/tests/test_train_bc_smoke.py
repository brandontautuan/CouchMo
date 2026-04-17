"""Smoke test: behavior cloning training on a tiny synthetic dataset.

Generates 50 random samples packed into one shard + manifest, runs 2 epochs,
and asserts the training loss decreases.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip entire module if torch not installed

from shared.dataset_format import EpisodeMeta, Manifest, write_shard  # noqa: E402
from training.imitation.train_bc import main  # noqa: E402


_N = 50
_RNG_SEED = 42
_FRAME_H, _FRAME_W = 84, 84  # native res (can be any; preprocess resizes anyway)


def _make_synthetic_shard(tmp_path: Path) -> Path:
    """Write a single episode with 50 random BGR frames + random actions."""
    rng = np.random.default_rng(_RNG_SEED)

    ep_dir = tmp_path / "ep_000"
    ep_dir.mkdir()

    lefts = rng.integers(0, 256, size=(_N, _FRAME_H, _FRAME_W, 3), dtype=np.uint8)
    rights = rng.integers(0, 256, size=(_N, _FRAME_H, _FRAME_W, 3), dtype=np.uint8)
    steers = rng.uniform(-1.0, 1.0, size=(_N,)).astype(np.float32)
    throttles = rng.uniform(0.0, 1.0, size=(_N,)).astype(np.float32)
    timestamps = np.arange(_N, dtype=np.float64) * 0.1

    shard_path = ep_dir / "shard_0.npz"
    write_shard(
        shard_path,
        left=lefts,
        right=rights,
        steer=steers,
        throttle=throttles,
        t=timestamps,
    )
    return ep_dir


def _write_manifest(tmp_path: Path) -> None:
    manifest = Manifest(
        episodes=[
            EpisodeMeta(
                id="ep_000",
                n_samples=_N,
                created_utc="2026-04-16T00:00:00Z",
                shard_path="ep_000",
            )
        ]
    )
    manifest.to_json(tmp_path / "manifest.json")


def test_bc_training_loss_decreases(tmp_path: Path) -> None:
    """Train for 2 epochs on 50 synthetic samples; assert loss goes down."""
    _make_synthetic_shard(tmp_path)
    _write_manifest(tmp_path)

    out_ckpt = tmp_path / "checkpoints" / "bc.pt"

    result = main([
        "--data-root", str(tmp_path),
        "--epochs", "5",
        "--batch-size", "16",
        "--device", "cpu",
        "--out", str(out_ckpt),
        "--val-split", "0.2",
        "--lr", "1e-3",
        "--num-workers", "0",
        "--seed", "0",
    ])

    train_losses = result["train_loss_history"]
    assert len(train_losses) == 5, f"Expected 5 epochs of loss, got {len(train_losses)}"

    initial_loss = train_losses[0]
    final_loss = train_losses[-1]
    assert final_loss < initial_loss, (
        f"Train loss did not decrease: {initial_loss:.6f} → {final_loss:.6f}"
    )

    # Checkpoint file was written.
    assert out_ckpt.exists(), "Checkpoint file was not created."

    # Meta JSON was written.
    meta_path = out_ckpt.with_suffix(".json").with_stem(out_ckpt.stem + "_meta")
    assert meta_path.exists(), "Meta JSON was not created."
