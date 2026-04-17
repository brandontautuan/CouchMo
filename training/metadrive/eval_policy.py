"""Evaluate an exported ONNX policy in MetaDrive and enforce go/no-go thresholds.

CLI::

    python -m training.metadrive.eval_policy \
        --onnx ./exports/couchmo_v1.onnx \
        --episodes 500

Exit codes:
    0 — all thresholds met (see GO_NO_GO below)
    1 — any threshold failed
    2 — runtime error

Thresholds from spec §R3.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Go/no-go thresholds (spec §R3).
GO_NO_GO = {
    "collision_rate_max": 0.05,
    "off_road_rate_max": 0.10,
    "mean_episode_length_min": 300.0,
}


def evaluate(
    onnx_path: Path,
    episodes: int,
    max_steps: int = 500,
    start_seed: int = 10_000,
) -> dict[str, Any]:
    """Run deterministic held-out eval and return a metrics dict."""
    if episodes <= 0:
        raise ValueError(f"episodes must be >= 1, got {episodes}")

    import onnxruntime as ort
    from training.metadrive.env import CouchMoMetaDriveEnv

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    env = CouchMoMetaDriveEnv(config={"num_scenarios": 2000, "randomize": False})

    collisions = 0
    off_roads = 0
    lengths: list[int] = []

    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=start_seed + ep)
            ep_len = 0
            for step in range(max_steps):
                x = obs[np.newaxis, :, :, :].astype(np.float32)
                action = sess.run(None, {input_name: x})[0][0]
                if not np.all(np.isfinite(action)):
                    raise RuntimeError(
                        f"Non-finite action from ONNX at ep={ep} step={step}: {action!r}"
                    )
                obs, _, term, trunc, _info = env.step(action)
                ep_len = step + 1
                if term or trunc:
                    # _is_collision / _is_off_road read vehicle state from
                    # self._md_env; no info arg (env.py:195-205).
                    if env._is_collision():
                        collisions += 1
                    elif env._is_off_road():
                        off_roads += 1
                    break
            # max_steps timeouts count toward mean_episode_length but are
            # classified as neither collision nor off-road (they inflate the
            # denominator for both rates, keeping a sluggish policy honest).
            lengths.append(ep_len)
            log.info("Episode %d/%d length=%d", ep + 1, episodes, ep_len)
    finally:
        env.close()

    return {
        "episodes": episodes,
        "collision_rate": collisions / episodes,
        "off_road_rate": off_roads / episodes,
        "mean_episode_length": float(np.mean(lengths)),
    }


def check_thresholds(report: dict[str, Any]) -> list[str]:
    """Return a list of human-readable failure strings (empty list = pass)."""
    failures: list[str] = []
    if report["collision_rate"] > GO_NO_GO["collision_rate_max"]:
        failures.append(
            f"collision_rate {report['collision_rate']:.3f} > "
            f"{GO_NO_GO['collision_rate_max']:.3f}"
        )
    if report["off_road_rate"] > GO_NO_GO["off_road_rate_max"]:
        failures.append(
            f"off_road_rate {report['off_road_rate']:.3f} > "
            f"{GO_NO_GO['off_road_rate_max']:.3f}"
        )
    if report["mean_episode_length"] < GO_NO_GO["mean_episode_length_min"]:
        failures.append(
            f"mean_episode_length {report['mean_episode_length']:.1f} < "
            f"{GO_NO_GO['mean_episode_length_min']:.1f}"
        )
    return failures


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate an exported ONNX policy in MetaDrive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--onnx", type=Path, required=True, help="Path to the exported ONNX.")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--start-seed", type=int, default=10_000, help="First eval seed (held out from training).")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    args = _build_parser().parse_args(argv)

    try:
        report = evaluate(
            onnx_path=args.onnx,
            episodes=args.episodes,
            max_steps=args.max_steps,
            start_seed=args.start_seed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    print(json.dumps(report, indent=2))

    failures = check_thresholds(report)
    if failures:
        print("GO/NO-GO FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    print("GO/NO-GO PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
