"""Round-trip smoke tests for ``shared.dataset_format``.

The shard format is a cross-surface contract (sim writer ↔ training reader).
These tests pin the key names, dtypes, and JSON manifest round-trip so any
silent rename or dtype drift breaks loudly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from shared.dataset_format import (
    SHARD_ARRAY_KEYS,
    EpisodeMeta,
    Manifest,
    read_shard,
    write_shard,
)


def _make_arrays(n: int = 3, h: int = 4, w: int = 4):
    rng = np.random.default_rng(0)
    return {
        "left": rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8),
        "right": rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8),
        "steer": rng.uniform(-1.0, 1.0, size=(n,)).astype(np.float32),
        "throttle": rng.uniform(0.0, 1.0, size=(n,)).astype(np.float32),
        "t": np.arange(n, dtype=np.float64) * 0.1,
    }


def test_shard_round_trip_preserves_all_keys(tmp_path: Path) -> None:
    arrays = _make_arrays()
    shard = tmp_path / "ep_000" / "shard_0.npz"

    write_shard(shard, **arrays)
    loaded = read_shard(shard)

    assert set(loaded.keys()) == set(SHARD_ARRAY_KEYS)
    for key in SHARD_ARRAY_KEYS:
        np.testing.assert_array_equal(loaded[key], arrays[key])
        assert loaded[key].dtype == arrays[key].dtype, f"{key} dtype changed"


def test_write_shard_rejects_mismatched_lengths(tmp_path: Path) -> None:
    import pytest

    arrays = _make_arrays(n=3)
    arrays["throttle"] = arrays["throttle"][:2]

    with pytest.raises(ValueError, match="leading dim"):
        write_shard(tmp_path / "bad.npz", **arrays)


def test_manifest_json_round_trip(tmp_path: Path) -> None:
    manifest = Manifest(
        episodes=[
            EpisodeMeta(
                id="ep_000",
                n_samples=100,
                created_utc="2026-04-15T12:00:00Z",
                shard_path="ep_000/shard_0.npz",
                world="synthetic_campus",
                notes="first run",
            ),
            EpisodeMeta(
                id="ep_001",
                n_samples=42,
                created_utc="2026-04-15T12:05:00Z",
            ),
        ]
    )
    path = tmp_path / "manifest.json"

    manifest.to_json(path)
    loaded = Manifest.from_json(path)

    assert loaded == manifest
