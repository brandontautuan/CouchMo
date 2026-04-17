"""Collect behavior cloning dataset by running MetaDrive's IDM expert.

CLI usage::

    python -m training.metadrive.collect_bc \
        --data-root ./training/data/metadrive_bc \
        --episodes 500 \
        --max-steps 500

Writes one shard per episode under ``<data-root>/episodes/<episode_id>/shard_0.npz``
plus a top-level ``manifest.json``. The output directory is drop-in compatible
with ``training.imitation.train_bc.load_dataset``.

Frames are stored as raw BGR uint8 (256x256x3 as produced by MetaDrive's
RGBCameras). Preprocessing + frame stacking is applied at training load time,
exactly as it is for simulation/-produced shards.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
import uuid
from pathlib import Path

import numpy as np

from shared.dataset_format import EpisodeMeta, Manifest, write_shard

log = logging.getLogger(__name__)


def collect(
    data_root: Path,
    episodes: int,
    max_steps: int,
    start_seed: int = 0,
) -> Path:
    """Roll out the IDM expert and write a BC dataset to ``data_root``.

    Returns the absolute data root path.
    """
    # Lazy imports — metadrive is a heavy optional dep.
    from training.metadrive.env import CouchMoMetaDriveEnv
    from training.metadrive.expert_policy import IDMExpertAdapter

    data_root = Path(data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    episodes_dir = data_root / "episodes"
    episodes_dir.mkdir(exist_ok=True)

    manifest = Manifest()

    env = CouchMoMetaDriveEnv(
        config={"num_scenarios": max(episodes, 100), "randomize": False}
    )
    try:
        for ep_idx in range(episodes):
            seed = start_seed + ep_idx
            env.reset(seed=seed)
            expert = IDMExpertAdapter(env)

            lefts: list[np.ndarray] = []
            rights: list[np.ndarray] = []
            steers: list[float] = []
            throttles: list[float] = []
            ts: list[float] = []

            for step in range(max_steps):
                # Read raw BGR for logging BEFORE preprocessing.
                left_bgr = env._read_camera_rgb("left_cam")
                right_bgr = env._read_camera_rgb("right_cam")

                action = expert.act()
                lefts.append(left_bgr)
                rights.append(right_bgr)
                steers.append(float(action[0]))
                throttles.append(float(action[1]))
                ts.append(step * 0.1)  # 10 Hz

                _, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break

            if not lefts:
                log.warning("Episode %d produced 0 samples; skipping.", ep_idx)
                continue

            ep_id = f"ep_{ep_idx:04d}_{uuid.uuid4().hex[:6]}"
            shard_rel = f"episodes/{ep_id}"
            shard_abs = data_root / shard_rel
            shard_abs.mkdir(parents=True, exist_ok=True)

            write_shard(
                shard_abs / "shard_0.npz",
                left=np.stack(lefts, axis=0),
                right=np.stack(rights, axis=0),
                steer=np.asarray(steers, dtype=np.float32),
                throttle=np.asarray(throttles, dtype=np.float32),
                t=np.asarray(ts, dtype=np.float32),
            )

            manifest.episodes.append(
                EpisodeMeta(
                    id=ep_id,
                    n_samples=len(lefts),
                    created_utc=_dt.datetime.utcnow().isoformat() + "Z",
                    shard_path=shard_rel,
                    world="metadrive_safe",
                    notes=f"seed={seed}",
                )
            )

            log.info("Episode %d/%d (seed=%d) -> %d samples", ep_idx + 1, episodes, seed, len(lefts))

        manifest.to_json(data_root / "manifest.json")
    finally:
        env.close()

    log.info("Wrote %d episodes to %s", len(manifest.episodes), data_root)
    return data_root


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect BC dataset from MetaDrive's IDM expert.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=Path, required=True, help="Output directory.")
    p.add_argument("--episodes", type=int, default=500, help="Number of episodes.")
    p.add_argument("--max-steps", type=int, default=500, help="Max steps per episode.")
    p.add_argument("--start-seed", type=int, default=0, help="First episode seed.")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    args = _build_parser().parse_args(argv)
    try:
        out = collect(
            data_root=args.data_root,
            episodes=args.episodes,
            max_steps=args.max_steps,
            start_seed=args.start_seed,
        )
        print(f"BC dataset written -> {out}")
    except Exception as exc:  # noqa: BLE001 — top-level CLI catch
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
