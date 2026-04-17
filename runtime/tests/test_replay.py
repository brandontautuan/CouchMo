"""End-to-end replay-mode test for ``python -m runtime.drive``.

Writes a 10-step synthetic episode shard with ``shared.dataset_format``,
invokes :func:`runtime.drive.main` in-process (faster and more hermetic than
subprocess), captures stdout, and asserts:

* the function returns 0
* stdout contains one line per shard step with both
  ``predicted=(...)`` and ``recorded=(...)`` tokens.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shared.dataset_format import write_shard


def _write_tiny_shard(path: Path, n: int = 10, h: int = 48, w: int = 64) -> None:
    rng = np.random.default_rng(0)
    write_shard(
        path,
        left=rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8),
        right=rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8),
        steer=rng.uniform(-1.0, 1.0, size=(n,)).astype(np.float32),
        throttle=rng.uniform(0.0, 1.0, size=(n,)).astype(np.float32),
        t=np.arange(n, dtype=np.float64) * 0.1,
    )


def test_replay_runs_and_prints_per_step_lines(
    tmp_path: Path,
    tiny_onnx_model: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shard = tmp_path / "ep_000" / "shard_0.npz"
    _write_tiny_shard(shard, n=10)

    from runtime.drive import main

    rc = main(
        [
            "--source", "replay",
            "--episode", str(shard),
            "--model", str(tiny_onnx_model),
            "--log-level", "WARNING",  # keep INFO banner out of stdout
        ]
    )
    assert rc == 0, f"drive.main returned non-zero exit code {rc}"

    captured = capsys.readouterr()
    # The per-step lines go to stdout (print), the log banner goes to stderr.
    step_lines = [
        ln for ln in captured.out.splitlines() if ln.startswith("step ")
    ]
    assert len(step_lines) == 10, (
        f"expected 10 step lines, got {len(step_lines)}\n"
        f"stdout was:\n{captured.out}"
    )
    for ln in step_lines:
        assert "predicted=" in ln, f"line missing predicted= token: {ln!r}"
        assert "recorded=" in ln, f"line missing recorded= token: {ln!r}"


def test_replay_missing_episode_exits_nonzero(
    tmp_path: Path,
    tiny_onnx_model: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from runtime.drive import main

    rc = main(
        [
            "--source", "replay",
            "--episode", str(tmp_path / "no_such_shard.npz"),
            "--model", str(tiny_onnx_model),
            "--log-level", "ERROR",
        ]
    )
    assert rc != 0
    # Error message goes to stderr.
    assert "not found" in capsys.readouterr().err
