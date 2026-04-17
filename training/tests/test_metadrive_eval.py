"""Tests for the eval_policy CLI."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# importorskip before hard torch import — collection must succeed on torchless hosts.
torch = pytest.importorskip("torch")
pytest.importorskip("metadrive")


def _make_dummy_onnx(path: Path, seed: int = 0) -> None:
    """Export a freshly-initialized BCPolicy to ONNX for use in smoke tests."""
    from training.imitation.export_onnx import export
    from training.imitation.model import BCPolicy

    torch.manual_seed(seed)
    bc = BCPolicy(in_channels=8)
    pt = path.with_suffix(".pt")
    torch.save({"state_dict": bc.state_dict(), "arch": {"in_channels": 8}}, pt)
    export(pt_path=pt, onnx_path=path, opset=17, verify=False)


def test_eval_policy_reports_metrics(tmp_path: Path):
    from training.metadrive.eval_policy import evaluate

    onnx = tmp_path / "model.onnx"
    _make_dummy_onnx(onnx)

    report = evaluate(onnx_path=onnx, episodes=2, max_steps=30, start_seed=10_000)
    for key in ("collision_rate", "off_road_rate", "mean_episode_length", "episodes"):
        assert key in report, f"missing key {key} in report: {report}"
    assert report["episodes"] == 2
    assert 0.0 <= report["collision_rate"] <= 1.0
    assert 0.0 <= report["off_road_rate"] <= 1.0
    assert report["mean_episode_length"] > 0
