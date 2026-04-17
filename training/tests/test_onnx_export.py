"""Numerical equivalence tests for ONNX export of BCPolicy (Task 8b).

Three test groups:
  1. test_equivalence_single_batch  — core equivalence, batch=3, max abs diff < 1e-4.
  2. test_dynamic_batch_consistency — batch=1 and batch=5 both produce matched outputs,
                                      confirming dynamic_axes are not silently dropped.
  3. test_onnx_structural_validity  — onnx.checker.check_model passes on the exported
                                      file (explicit structural guard, even though export()
                                      calls it unconditionally).

All tests operate on a synthetically trained model (1 optimizer step on random data) to
exercise a non-zero, non-init state_dict that can surface ONNX graph subtleties hidden
by init weights.  No real data, no manifest, no shards needed.

Target runtime: < 5 s per test on CPU.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

# Guard at import time — the three packages must all be present.
torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from training.imitation.model import BCPolicy
from training.imitation.export_onnx import export


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_trained_checkpoint(tmp_path, seed: int = 42) -> tuple:
    """Build a 1-step trained BCPolicy, save in Task-8a format, return (model, pt_path)."""
    torch.manual_seed(seed)
    model = BCPolicy(in_channels=8)

    # One optimizer step on random data so state_dict is non-trivially different
    # from freshly-initialized weights.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x = torch.rand(2, 8, 84, 84)
    targets = torch.rand(2, 2)
    # Clamp targets into valid ranges for steer/throttle (mirrors real label bounds).
    targets[:, 0] = targets[:, 0] * 2 - 1   # steer in [-1, 1]
    targets[:, 1] = targets[:, 1]             # throttle in [0, 1]

    model.train()
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, targets)
    loss.backward()
    optimizer.step()

    # Save in Task-8a checkpoint format.
    pt_path = tmp_path / "bc.pt"
    torch.save({"state_dict": model.state_dict(), "arch": {"in_channels": 8}}, pt_path)

    model.eval()
    return model, pt_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_equivalence_single_batch(tmp_path):
    """Core test: Torch and ONNX Runtime agree to within 1e-4 on batch=3 input."""
    model, pt_path = _make_trained_checkpoint(tmp_path)
    onnx_path = tmp_path / "bc.onnx"

    returned_path = export(pt_path, onnx_path=onnx_path)
    assert returned_path == onnx_path
    assert onnx_path.exists(), "export() did not write the .onnx file"

    # Run inference.
    torch.manual_seed(0)
    x = torch.rand(3, 8, 84, 84)

    with torch.no_grad():
        y_torch = model(x).numpy()

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    y_onnx = sess.run(None, {"input": x.numpy()})[0]

    assert y_torch.shape == y_onnx.shape, (
        f"Shape mismatch: Torch={y_torch.shape}, ONNX={y_onnx.shape}"
    )

    max_diff = float(np.max(np.abs(y_torch - y_onnx)))
    assert max_diff < 1e-4, (
        f"Max abs diff {max_diff:.2e} exceeds 1e-4 threshold — ONNX graph disagrees with Torch"
    )


def test_dynamic_batch_consistency(tmp_path):
    """Dynamic axes guard: batch=1 and batch=5 both run and match Torch outputs."""
    model, pt_path = _make_trained_checkpoint(tmp_path, seed=7)
    onnx_path = tmp_path / "bc_dyn.onnx"

    export(pt_path, onnx_path=onnx_path)

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )

    for batch_size in (1, 5):
        torch.manual_seed(batch_size)
        x = torch.rand(batch_size, 8, 84, 84)

        with torch.no_grad():
            y_torch = model(x).numpy()

        y_onnx = sess.run(None, {"input": x.numpy()})[0]

        assert y_torch.shape == (batch_size, 2), (
            f"Torch output shape wrong for batch={batch_size}: {y_torch.shape}"
        )
        assert y_onnx.shape == (batch_size, 2), (
            f"ONNX output shape wrong for batch={batch_size}: {y_onnx.shape}"
        )

        max_diff = float(np.max(np.abs(y_torch - y_onnx)))
        assert max_diff < 1e-4, (
            f"batch={batch_size}: max abs diff {max_diff:.2e} >= 1e-4"
        )


def test_onnx_structural_validity(tmp_path):
    """Explicit structural guard: the exported ONNX file must pass onnx.checker.check_model."""
    _, pt_path = _make_trained_checkpoint(tmp_path, seed=99)
    onnx_path = tmp_path / "bc_valid.onnx"

    export(pt_path, onnx_path=onnx_path)

    # Load and check independently of what export() already did internally.
    onnx_model = onnx.load(str(onnx_path))
    # Raises onnx.checker.ValidationError if the graph is malformed.
    onnx.checker.check_model(onnx_model)

    # Basic opset sanity: we requested opset 17 (default).
    opset_versions = [op.version for op in onnx_model.opset_import]
    assert 17 in opset_versions, (
        f"Expected opset 17 in exported model; found: {opset_versions}"
    )

    # Output shape annotation should exist and end in 2 (steer + throttle).
    graph_outputs = onnx_model.graph.output
    assert len(graph_outputs) == 1, f"Expected 1 output, got {len(graph_outputs)}"
    type_proto = graph_outputs[0].type.tensor_type
    shape = type_proto.shape
    assert shape is not None, "ONNX output has no shape annotation"
    dims = [d.dim_value for d in shape.dim]
    assert dims[-1] == 2, f"Expected last output dim=2 (steer, throttle); got dims={dims}"
