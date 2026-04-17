"""Tests for the pure-Python ``ShardBuffer`` class in ``dataset_recorder_node``.

These tests exercise the shard-buffer logic in isolation — no ROS, no
``rclpy``, no Gazebo.  They import only ``ShardBuffer`` (pure-Python path in
the recorder module) and the ``shared.dataset_format`` read/write helpers.

Shard-path naming convention used by ``ShardBuffer``:
    ``<episode_dir>/shard_<n>.npz``  (plain integer, zero-indexed, not
    zero-padded — e.g. ``shard_0.npz``, ``shard_1.npz``, …)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the recorder module is importable on bare hosts by pointing sys.path
# to the sim package.  The conftest.py already adds the repo root so
# ``shared`` is available; we also add the ROS package source tree.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIM_PKG = REPO_ROOT / "simulation" / "src" / "couchmo_expert"
if str(SIM_PKG) not in sys.path:
    sys.path.insert(0, str(SIM_PKG))

from couchmo_expert.dataset_recorder_node import ShardBuffer  # noqa: E402
from shared.dataset_format import read_shard  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(h: int = 4, w: int = 4, seed: int = 0) -> np.ndarray:
    """Return a random (H, W, 3) uint8 BGR frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _add_sample(buf: ShardBuffer, t: float = 0.0, seed: int = 0) -> Path | None:
    """Helper: add one sample with deterministic data."""
    return buf.add(
        left=_make_frame(seed=seed),
        right=_make_frame(seed=seed + 1000),
        steer=float(np.random.default_rng(seed).uniform(-1, 1)),
        throttle=float(np.random.default_rng(seed).uniform(0, 1)),
        t=t,
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_invalid_samples_per_shard_raises() -> None:
    """samples_per_shard < 1 should raise ValueError on construction."""
    with pytest.raises(ValueError, match="samples_per_shard must be >= 1"):
        ShardBuffer("/tmp/ep_test", samples_per_shard=0)


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------


def test_buffer_accumulates_without_flushing(tmp_path: Path) -> None:
    """Samples below the threshold must not trigger an automatic flush."""
    buf = ShardBuffer(tmp_path / "ep_acc", samples_per_shard=5)
    for i in range(4):
        result = _add_sample(buf, t=float(i), seed=i)
        assert result is None, "should not flush before threshold"
    assert buf.buffered_samples == 4
    assert buf.total_samples == 0
    assert buf.shard_index == 0


def test_buffer_flushes_at_threshold(tmp_path: Path) -> None:
    """The Nth sample must trigger a flush and return the shard path."""
    N = 3
    ep_dir = tmp_path / "ep_flush"
    buf = ShardBuffer(ep_dir, samples_per_shard=N)
    result = None
    for i in range(N):
        result = _add_sample(buf, t=float(i), seed=i)

    assert result is not None, "flush must return shard path on Nth sample"
    assert result == ep_dir / "shard_0.npz"
    assert result.is_file(), "shard file must exist on disk"
    assert buf.buffered_samples == 0, "buffer must be cleared after flush"
    assert buf.total_samples == N
    assert buf.shard_index == 1  # next shard will be shard_1


def test_multiple_auto_flushes(tmp_path: Path) -> None:
    """Two full shards must produce shard_0 and shard_1 on disk."""
    N = 2
    ep_dir = tmp_path / "ep_multi"
    buf = ShardBuffer(ep_dir, samples_per_shard=N)

    # First shard
    for i in range(N):
        _add_sample(buf, t=float(i), seed=i)
    assert (ep_dir / "shard_0.npz").is_file()
    assert buf.shard_index == 1

    # Second shard
    for i in range(N):
        _add_sample(buf, t=float(N + i), seed=N + i)
    assert (ep_dir / "shard_1.npz").is_file()
    assert buf.shard_index == 2
    assert buf.total_samples == 2 * N


# ---------------------------------------------------------------------------
# Manual flush (partial shard)
# ---------------------------------------------------------------------------


def test_explicit_flush_writes_partial_shard(tmp_path: Path) -> None:
    """flush() on a non-empty buffer must write a shard and clear the buffer."""
    ep_dir = tmp_path / "ep_partial"
    buf = ShardBuffer(ep_dir, samples_per_shard=10)
    for i in range(3):
        _add_sample(buf, t=float(i), seed=i)

    shard_path = buf.flush()
    assert shard_path is not None
    assert shard_path == ep_dir / "shard_0.npz"
    assert shard_path.is_file()
    assert buf.buffered_samples == 0
    assert buf.total_samples == 3


def test_explicit_flush_on_empty_buffer_returns_none(tmp_path: Path) -> None:
    """flush() on an already-empty buffer must return None without error."""
    buf = ShardBuffer(tmp_path / "ep_empty", samples_per_shard=5)
    result = buf.flush()
    assert result is None
    assert buf.total_samples == 0


def test_flush_after_auto_flush_increments_shard_index(tmp_path: Path) -> None:
    """Manual flush after an auto-flush must produce shard_1, not overwrite shard_0."""
    ep_dir = tmp_path / "ep_seq"
    buf = ShardBuffer(ep_dir, samples_per_shard=2)
    # Trigger auto-flush (shard_0)
    for i in range(2):
        _add_sample(buf, t=float(i), seed=i)
    assert buf.shard_index == 1

    # Add one more sample, then manual flush → shard_1
    _add_sample(buf, t=2.0, seed=99)
    shard_path = buf.flush()
    assert shard_path == ep_dir / "shard_1.npz"
    assert shard_path.is_file()
    assert buf.shard_index == 2


# ---------------------------------------------------------------------------
# Shard naming convention
# ---------------------------------------------------------------------------


def test_shard_paths_are_plain_integers_zero_indexed(tmp_path: Path) -> None:
    """Shard files must be named shard_0.npz, shard_1.npz (plain int, no padding)."""
    ep_dir = tmp_path / "ep_names"
    buf = ShardBuffer(ep_dir, samples_per_shard=1)
    for i in range(3):
        path = _add_sample(buf, t=float(i), seed=i)
        assert path == ep_dir / f"shard_{i}.npz"


# ---------------------------------------------------------------------------
# Round-trip: buffer → write_shard → read_shard
# ---------------------------------------------------------------------------


def test_round_trip_data_matches(tmp_path: Path) -> None:
    """Data written by ShardBuffer must survive a read_shard round-trip."""
    ep_dir = tmp_path / "ep_roundtrip"
    N = 4
    buf = ShardBuffer(ep_dir, samples_per_shard=N)

    rng = np.random.default_rng(42)
    lefts = [rng.integers(0, 256, (4, 6, 3), dtype=np.uint8) for _ in range(N)]
    rights = [rng.integers(0, 256, (4, 6, 3), dtype=np.uint8) for _ in range(N)]
    steers = rng.uniform(-1, 1, N).tolist()
    throttles = rng.uniform(0, 1, N).tolist()
    ts = [float(i) * 0.1 for i in range(N)]

    for i in range(N):
        buf.add(left=lefts[i], right=rights[i], steer=steers[i], throttle=throttles[i], t=ts[i])

    shard_path = ep_dir / "shard_0.npz"
    assert shard_path.is_file(), "shard must be written when buffer reaches N"

    loaded = read_shard(shard_path)

    np.testing.assert_array_equal(loaded["left"], np.stack(lefts))
    np.testing.assert_array_equal(loaded["right"], np.stack(rights))
    np.testing.assert_array_almost_equal(loaded["steer"], np.array(steers, dtype=np.float32))
    np.testing.assert_array_almost_equal(loaded["throttle"], np.array(throttles, dtype=np.float32))
    np.testing.assert_array_almost_equal(loaded["t"], np.array(ts, dtype=np.float64))

    assert loaded["left"].dtype == np.uint8
    assert loaded["right"].dtype == np.uint8
    assert loaded["steer"].dtype == np.float32
    assert loaded["throttle"].dtype == np.float32
    assert loaded["t"].dtype == np.float64


def test_partial_flush_round_trip(tmp_path: Path) -> None:
    """Partial shard written by explicit flush() must also round-trip correctly."""
    ep_dir = tmp_path / "ep_partial_rt"
    buf = ShardBuffer(ep_dir, samples_per_shard=10)

    rng = np.random.default_rng(7)
    n = 3
    lefts = [rng.integers(0, 256, (2, 2, 3), dtype=np.uint8) for _ in range(n)]
    rights = [rng.integers(0, 256, (2, 2, 3), dtype=np.uint8) for _ in range(n)]

    for i in range(n):
        buf.add(left=lefts[i], right=rights[i], steer=0.0, throttle=0.5, t=float(i))

    shard_path = buf.flush()
    assert shard_path is not None
    loaded = read_shard(shard_path)
    assert len(loaded["left"]) == n
    np.testing.assert_array_equal(loaded["left"], np.stack(lefts))
