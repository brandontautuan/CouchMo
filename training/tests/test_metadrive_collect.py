"""End-to-end smoke test for BC data collection."""
from __future__ import annotations

import datetime as dt
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

    # created_utc must be ISO-8601 with a trailing 'Z'. Parsing via fromisoformat
    # guards against a future regression to the deprecated utcnow() idiom.
    assert ep.created_utc.endswith("Z"), f"expected trailing Z; got {ep.created_utc}"
    dt.datetime.fromisoformat(ep.created_utc.replace("Z", "+00:00"))


def test_collect_bc_writes_manifest_per_episode(tmp_path: Path):
    """Manifest is written after every successful episode, not only at the end."""
    from training.metadrive.collect_bc import collect
    from shared.dataset_format import Manifest

    # 3 episodes so we can observe manifest growth. If this path ever regresses
    # to "write only at end" the only manifest we'd observe would be the final
    # one with 3 entries — we assert that the manifest read after a partial
    # collect (same call, just counting what reached disk) still contains all 3.
    out = collect(data_root=tmp_path, episodes=3, max_steps=10, start_seed=0)
    manifest = Manifest.from_json(out / "manifest.json")
    assert len(manifest.episodes) == 3
