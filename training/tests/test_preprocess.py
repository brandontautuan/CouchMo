"""Tests for ``shared.preprocess``.

These tests pin the behavior every downstream surface (sim writer, training
loader, runtime inference) depends on:

* ``preprocess_pair`` is a deterministic, uint8, (2, 84, 84) reducer.
* ``FrameStacker`` yields a fixed-shape float32 tensor in [0, 1] from the
  first push onward, and rolls oldest-out / newest-in.
"""
from __future__ import annotations

import numpy as np

from shared.preprocess import FrameStacker, preprocess_pair


def _make_frame(seed: int, h: int = 480, w: int = 640) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _make_uniform_pair(value: int) -> np.ndarray:
    pair = np.full((2, 84, 84), value, dtype=np.uint8)
    return pair


def test_preprocess_pair_shape_dtype() -> None:
    left = _make_frame(seed=0)
    right = _make_frame(seed=1)

    out = preprocess_pair(left, right)

    assert out.shape == (2, 84, 84), f"unexpected shape {out.shape}"
    assert out.dtype == np.uint8, f"unexpected dtype {out.dtype}"


def test_framestacker_first_push_repeats() -> None:
    stacker = FrameStacker(num_frames=4)
    pair = _make_uniform_pair(value=123)

    obs = stacker.push(pair)

    assert obs.shape == (8, 84, 84)
    assert obs.dtype == np.float32
    assert float(obs.min()) >= 0.0
    assert float(obs.max()) <= 1.0

    expected_value = np.float32(123) / np.float32(255.0)
    np.testing.assert_allclose(obs, np.full_like(obs, expected_value))

    for ch in range(0, 8, 2):
        np.testing.assert_array_equal(obs[ch], obs[0])
        np.testing.assert_array_equal(obs[ch + 1], obs[1])


def test_framestacker_rolls() -> None:
    stacker = FrameStacker(num_frames=4)

    values = [10, 50, 90, 130, 170]
    last_obs: np.ndarray | None = None
    for v in values:
        last_obs = stacker.push(_make_uniform_pair(v))

    assert last_obs is not None
    assert last_obs.shape == (8, 84, 84)

    head_value = float(last_obs[0, 0, 0])
    assert np.isclose(head_value, 170 / 255.0), f"newest frame should head the stack; got {head_value}"

    flat_values = {float(last_obs[ch, 0, 0]) for ch in range(0, 8, 2)}
    expected_kept = {v / 255.0 for v in values[-4:]}
    for expected in expected_kept:
        assert any(np.isclose(expected, fv) for fv in flat_values), (
            f"expected kept value {expected} not found in channel heads {flat_values}"
        )

    dropped = 10 / 255.0
    assert not any(np.isclose(dropped, fv) for fv in flat_values), (
        f"oldest frame {dropped} should have been dropped but is still in {flat_values}"
    )


def test_preprocess_deterministic() -> None:
    left = _make_frame(seed=42)
    right = _make_frame(seed=43)

    first = preprocess_pair(left, right)
    second = preprocess_pair(left, right)

    assert first.tobytes() == second.tobytes(), "preprocess_pair must be byte-deterministic"


def test_preprocess_rejects_float_input() -> None:
    import pytest

    left = _make_frame(seed=0).astype(np.float32) / 255.0
    right = _make_frame(seed=1)

    with pytest.raises(ValueError, match="uint8"):
        preprocess_pair(left, right)


def test_framestacker_first_push_isolated_from_caller_mutation() -> None:
    stacker = FrameStacker(num_frames=4)
    pair = _make_uniform_pair(value=100)

    stacker.push(pair)
    pair[:] = 0
    obs = stacker.push(_make_uniform_pair(value=200))

    expected_kept = np.float32(100) / np.float32(255.0)
    np.testing.assert_allclose(obs[2], np.full_like(obs[2], expected_kept))
