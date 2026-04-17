"""End-to-end smoke test for BC data collection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("metadrive")
pytest.importorskip("gymnasium")


def test_collect_bc_writes_valid_shards(tmp_path: Path):
    from training.metadrive.collect_bc import collect

    out = collect(
        data_root=tmp_path,
        episodes=2,
        max_steps=30,
        start_seed=0,
    )

    manifest_path = out / "manifest.json"
    assert manifest_path.exists(), "manifest.json not written"

    from shared.dataset_format import Manifest, read_shard

    manifest = Manifest.from_json(manifest_path)
    assert len(manifest.episodes) == 2, f"expected 2 episodes; got {len(manifest.episodes)}"

    ep = manifest.episodes[0]
    shard_dir = tmp_path / ep.shard_path
    shard_files = sorted(shard_dir.glob("shard_*.npz"))
    assert shard_files, f"no shards in {shard_dir}"

    shard = read_shard(shard_files[0])
    assert shard["left"].ndim == 4 and shard["left"].shape[-1] == 3
    assert shard["right"].ndim == 4 and shard["right"].shape[-1] == 3
    assert shard["left"].dtype == np.uint8
    assert shard["steer"].dtype == np.float32
    assert shard["throttle"].dtype == np.float32
    n = len(shard["left"])
    assert (
        len(shard["right"]) == n
        and len(shard["steer"]) == n
        and len(shard["throttle"]) == n
        and len(shard["t"]) == n
    )
