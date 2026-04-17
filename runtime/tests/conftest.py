"""Pytest bootstrap and shared fixtures for the runtime test suite.

* Prepends the repo root to ``sys.path`` so ``shared.*`` and
  ``serial_controller`` import without installing anything.
* Provides a session-scoped ``tiny_onnx_model`` fixture that builds a minimal
  BCPolicy-shaped ONNX model using PyTorch.  PyTorch is a DEV-time dependency
  only; the runtime itself must not import it.  Guarded with
  ``pytest.importorskip("torch")`` so hosts without torch skip cleanly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="session")
def tiny_onnx_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export a tiny ONNX model with the BCPolicy I/O contract.

    Input:  ``(batch, 8, 84, 84)`` float32, named ``"input"``.
    Output: ``(batch, 2)`` float32.

    The model is random — outputs are NOT meaningful, but the graph shape
    and dtype are exactly what :class:`runtime.inference.Policy` expects.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class _TinyPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # 8 input channels → pool to (1, 1) → 2 outputs.  Tiny and cheap.
            self.conv = nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(4, 2)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = self.conv(x)
            h = self.pool(h).flatten(start_dim=1)
            out = self.fc(h)
            steer = torch.tanh(out[:, 0:1])
            throttle = torch.sigmoid(out[:, 1:2])
            return torch.cat([steer, throttle], dim=1)

    torch.manual_seed(0)
    model = _TinyPolicy().eval()

    tmp = tmp_path_factory.mktemp("runtime_tiny_onnx")
    onnx_path = tmp / "tiny.onnx"

    dummy = torch.zeros(1, 8, 84, 84)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )
    return onnx_path
