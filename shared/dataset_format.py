"""On-disk dataset format used by simulation (writer) and training (reader).

Each episode is one ``.npz`` shard plus an entry in a top-level ``manifest.json``.
Pure stdlib + numpy. No torch, no ROS, no OpenCV.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SHARD_ARRAY_KEYS: tuple[str, ...] = ("left", "right", "steer", "throttle", "t")


def write_shard(
    path: Path,
    *,
    left: np.ndarray,
    right: np.ndarray,
    steer: np.ndarray,
    throttle: np.ndarray,
    t: np.ndarray,
) -> None:
    """Write a single episode shard as a compressed ``.npz`` file.

    Arrays must all have the same leading (sample) dimension. ``left`` and
    ``right`` are expected to be ``(N, H, W, 3)`` uint8 BGR frames but this
    function does not enforce shapes beyond "same length"; the training loader
    is the source of truth for expected shapes.
    """
    path = Path(path)
    lengths = {
        "left": len(left),
        "right": len(right),
        "steer": len(steer),
        "throttle": len(throttle),
        "t": len(t),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All arrays must share leading dim; got {lengths}")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        left=left,
        right=right,
        steer=steer,
        throttle=throttle,
        t=t,
    )


def read_shard(path: Path) -> dict[str, np.ndarray]:
    """Read an episode shard written by :func:`write_shard`."""
    path = Path(path)
    with np.load(path) as data:
        out = {key: data[key] for key in SHARD_ARRAY_KEYS}
    return out


@dataclass
class EpisodeMeta:
    """Metadata describing one recorded episode shard."""

    id: str
    n_samples: int
    created_utc: str
    shard_path: str = ""
    world: str = ""
    notes: str = ""


@dataclass
class Manifest:
    """Top-level manifest listing every episode shard in a dataset root."""

    episodes: list[EpisodeMeta] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        """Serialize the manifest to ``path`` as pretty-printed JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"episodes": [asdict(ep) for ep in self.episodes]}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "Manifest":
        """Load a manifest previously written by :meth:`to_json`."""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes = [EpisodeMeta(**entry) for entry in payload.get("episodes", [])]
        return cls(episodes=episodes)
