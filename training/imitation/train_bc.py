"""Behavior cloning training loop for CouchMo.

CLI usage::

    python -m training.imitation.train_bc \
        --data-root ./data \
        --epochs 10 \
        --batch-size 64 \
        --device cpu \
        --out ./checkpoints/bc.pt

No ROS. No Gazebo. Pure torch + numpy + OpenCV (via shared.preprocess).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from shared.dataset_format import Manifest
from shared.preprocess import FrameStacker, preprocess_pair
from training.imitation.model import BCPolicy

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

BC_META_SCHEMA: str = "bc_meta/v1"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all shards under *data_root* into contiguous numpy arrays.

    Applies ``preprocess_pair`` + ``FrameStacker`` at load time.  Episode
    boundaries reset the frame stacker so frames never bleed across episodes.

    Returns:
        obs   : (N, 8, 84, 84) float32 in [0, 1].
        targets: (N, 2) float32 — column 0 = steer, column 1 = throttle.
    """
    data_root = Path(data_root)
    manifest_path = data_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}")

    manifest = Manifest.from_json(manifest_path)
    if not manifest.episodes:
        raise ValueError(f"Manifest at {manifest_path} contains no episodes.")

    all_obs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for ep_meta in manifest.episodes:
        ep_dir = data_root / ep_meta.shard_path
        shard_paths = sorted(
            ep_dir.glob("shard_*.npz"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not shard_paths:
            log.warning("No shards found in %s — skipping.", ep_dir)
            continue

        stacker = FrameStacker(num_frames=4)

        for shard_path in shard_paths:
            with np.load(shard_path) as data:
                lefts = data["left"]       # (n, H, W, 3) uint8
                rights = data["right"]     # (n, H, W, 3) uint8
                steers = data["steer"]     # (n,) float32
                throttles = data["throttle"]  # (n,) float32

            for i in range(len(lefts)):
                pair = preprocess_pair(lefts[i], rights[i])
                obs = stacker.push(pair)
                all_obs.append(obs)
                all_targets.append(np.array([steers[i], throttles[i]], dtype=np.float32))

    if not all_obs:
        raise ValueError(f"No samples loaded from {data_root}.")

    # TODO(scalability): load_dataset() materialises the full dataset in
    # memory as one (N, 8, 84, 84) float32 array — ~226 KB/sample. At
    # N >= ~50k samples this will exceed 10 GB RAM on typical laptops.
    # Revisit with a streaming IterableDataset once real campus
    # recordings exceed that volume.
    obs_arr = np.stack(all_obs, axis=0)          # (N, 8, 84, 84)
    targets_arr = np.stack(all_targets, axis=0)  # (N, 2)
    return obs_arr, targets_arr


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _make_split(
    n_total: int, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/val index split without shuffle."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_val = max(1, int(n_total * val_frac))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return train_idx, val_idx


def _run_epoch(
    model: BCPolicy,
    loader: DataLoader,
    criterion: nn.MSELoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    """Run one pass (train or val) and return loss + MAE metrics."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_steer_mae = 0.0
    total_throttle_mae = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for obs_batch, target_batch in loader:
            obs_batch = obs_batch.to(device)
            target_batch = target_batch.to(device)

            pred = model(obs_batch)
            loss = criterion(pred, target_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_steer_mae += (pred[:, 0] - target_batch[:, 0]).abs().mean().item()
            total_throttle_mae += (pred[:, 1] - target_batch[:, 1]).abs().mean().item()
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "steer_mae": total_steer_mae / n_batches,
        "throttle_mae": total_throttle_mae / n_batches,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Behavior cloning training for CouchMo.")
    p.add_argument("--data-root", type=Path, default=Path("./data"),
                   help="Root directory containing manifest.json and episode shards.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out", type=Path, default=Path("./checkpoints/bc.pt"))
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> dict:
    """Entry point — callable from tests by passing *argv* list.

    Returns:
        A dict with ``train_loss_history``, ``val_loss_history``, and
        ``final_metrics`` so tests can inspect results without touching the
        filesystem.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    log.info("Loading dataset from %s …", args.data_root)
    t0 = time.perf_counter()
    obs, targets = load_dataset(args.data_root)
    log.info("Loaded %d samples in %.1fs.", len(obs), time.perf_counter() - t0)

    n_total = len(obs)
    train_idx, val_idx = _make_split(n_total, args.val_split, args.seed)
    n_train = len(train_idx)
    n_val = len(val_idx)
    log.info("Split: %d train / %d val.", n_train, n_val)

    obs_t = torch.from_numpy(obs)
    targets_t = torch.from_numpy(targets)

    train_ds = TensorDataset(obs_t[train_idx], targets_t[train_idx])
    val_ds = TensorDataset(obs_t[val_idx], targets_t[val_idx])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Model + optimiser
    # ------------------------------------------------------------------
    model = BCPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    train_steer_mae_history: list[float] = []
    train_throttle_mae_history: list[float] = []
    val_steer_mae_history: list[float] = []
    val_throttle_mae_history: list[float] = []

    for epoch in range(1, args.epochs + 1):
        t_ep = time.perf_counter()

        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = _run_epoch(model, val_loader, criterion, None, device)

        train_loss_history.append(train_metrics["loss"])
        val_loss_history.append(val_metrics["loss"])
        train_steer_mae_history.append(train_metrics["steer_mae"])
        train_throttle_mae_history.append(train_metrics["throttle_mae"])
        val_steer_mae_history.append(val_metrics["steer_mae"])
        val_throttle_mae_history.append(val_metrics["throttle_mae"])

        log.info(
            "Epoch %d/%d  [%.1fs]  "
            "train_loss=%.4f  val_loss=%.4f  "
            "train_steer_mae=%.4f  train_throttle_mae=%.4f  "
            "val_steer_mae=%.4f  val_throttle_mae=%.4f",
            epoch,
            args.epochs,
            time.perf_counter() - t_ep,
            train_metrics["loss"],
            val_metrics["loss"],
            train_metrics["steer_mae"],
            train_metrics["throttle_mae"],
            val_metrics["steer_mae"],
            val_metrics["throttle_mae"],
        )

    # ------------------------------------------------------------------
    # Save checkpoint + meta
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": {"in_channels": 8},
        },
        out_path,
    )
    log.info("Checkpoint saved → %s", out_path)

    final_metrics = {
        "train_loss": train_loss_history[-1],
        "val_loss": val_loss_history[-1],
        "train_steer_mae": train_steer_mae_history[-1],
        "train_throttle_mae": train_throttle_mae_history[-1],
        "val_steer_mae": val_steer_mae_history[-1],
        "val_throttle_mae": val_throttle_mae_history[-1],
    }

    meta = {
        "schema": BC_META_SCHEMA,
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "metrics": {
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "train_steer_mae_history": train_steer_mae_history,
            "train_throttle_mae_history": train_throttle_mae_history,
            "val_steer_mae_history": val_steer_mae_history,
            "val_throttle_mae_history": val_throttle_mae_history,
            "final_metrics": final_metrics,
        },
        "num_samples": n_total,
        "num_train": n_train,
        "num_val": n_val,
    }

    meta_path = out_path.with_suffix(".json").with_stem(out_path.stem + "_meta")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Meta saved → %s", meta_path)

    return {
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "final_metrics": final_metrics,
    }


if __name__ == "__main__":
    main()
