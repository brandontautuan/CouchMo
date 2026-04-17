"""Smoke tests for ``runtime.inference.Policy``.

Uses a tiny ONNX model built at fixture time (PyTorch dev-time only — the
runtime must not import torch).  Confirms that :class:`Policy` loads the
model, runs inference from stacked input and from a raw stereo BGR pair, and
returns a ``(steer, throttle)`` tuple of plain Python floats.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from runtime.inference import Policy


# Bounds are deliberately loose — the fixture model is random, so exact
# values are uninformative.  We only guard against NaN / infinite / wildly-
# out-of-range outputs that would indicate a broken graph or dtype mismatch.
_STEER_LOW, _STEER_HIGH = -2.0, 2.0
_THROTTLE_LOW, _THROTTLE_HIGH = -1.0, 2.0


def test_policy_loads_from_path(tiny_onnx_model: Path) -> None:
    """Smoke: Policy constructs from a Path and from a str path."""
    p_path = Policy(tiny_onnx_model)
    p_str = Policy(str(tiny_onnx_model))
    assert p_path is not None
    assert p_str is not None


def test_policy_missing_model_raises(tmp_path: Path) -> None:
    """FileNotFoundError is raised for a path that does not exist."""
    import pytest

    with pytest.raises(FileNotFoundError):
        Policy(tmp_path / "does_not_exist.onnx")


def test_predict_from_stacked_returns_float_tuple(tiny_onnx_model: Path) -> None:
    """Feeding a (8, 84, 84) stacked tensor yields Python-float (steer, throttle)."""
    policy = Policy(tiny_onnx_model)
    rng = np.random.default_rng(42)
    stacked = rng.random((8, 84, 84), dtype=np.float32)

    result = policy.predict_from_stacked(stacked)

    assert isinstance(result, tuple) and len(result) == 2
    steer, throttle = result
    assert type(steer) is float, f"steer type was {type(steer).__name__}"
    assert type(throttle) is float, f"throttle type was {type(throttle).__name__}"
    assert _STEER_LOW <= steer <= _STEER_HIGH, f"steer={steer} out of loose range"
    assert _THROTTLE_LOW <= throttle <= _THROTTLE_HIGH, (
        f"throttle={throttle} out of loose range"
    )


def test_predict_from_pair_uses_framestacker(tiny_onnx_model: Path) -> None:
    """Feeding a (2, 84, 84) preprocessed pair runs through the internal stacker."""
    policy = Policy(tiny_onnx_model)
    rng = np.random.default_rng(7)
    pair = rng.integers(0, 256, size=(2, 84, 84), dtype=np.uint8)

    steer, throttle = policy.predict_from_pair(pair)

    assert isinstance(steer, float) and isinstance(throttle, float)
    assert _STEER_LOW <= steer <= _STEER_HIGH
    assert _THROTTLE_LOW <= throttle <= _THROTTLE_HIGH


def test_predict_from_raw_bgr_pair(tiny_onnx_model: Path) -> None:
    """Full pipeline: raw (H,W,3) uint8 BGR frames → preprocess → stack → infer."""
    policy = Policy(tiny_onnx_model)
    rng = np.random.default_rng(99)
    # Use arbitrary H, W to exercise the resize path in preprocess_pair.
    left = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    right = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)

    steer, throttle = policy.predict(left, right)

    assert isinstance(steer, float) and isinstance(throttle, float)
    assert _STEER_LOW <= steer <= _STEER_HIGH
    assert _THROTTLE_LOW <= throttle <= _THROTTLE_HIGH


def test_reset_clears_stacker(tiny_onnx_model: Path) -> None:
    """Policy.reset() drops the frame history without breaking subsequent calls."""
    policy = Policy(tiny_onnx_model)
    rng = np.random.default_rng(1)
    pair = rng.integers(0, 256, size=(2, 84, 84), dtype=np.uint8)

    policy.predict_from_pair(pair)
    policy.reset()
    # Next call must still succeed (i.e. stacker re-primes from this frame).
    steer, throttle = policy.predict_from_pair(pair)
    assert isinstance(steer, float) and isinstance(throttle, float)
