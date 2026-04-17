"""On-laptop inference loop for CouchMo.

This module is the deploy target: it glues :class:`runtime.inference.Policy`,
:class:`runtime.camera.DualCamera`, and :class:`runtime.control.Controller`
together into a monotonic 10 Hz control loop, and adds a sim-free replay mode
that scores a recorded episode shard so sanity can be verified on any laptop
without hardware.

CLI summary::

    # Live mode — real cameras + serial out
    python -m runtime.drive --source live \\
        --left-cam 0 --right-cam 1 \\
        --port COM3 --model model.onnx --rate 10

    # Replay mode — read an .npz episode, run Policy, print predicted vs recorded
    python -m runtime.drive --source replay \\
        --episode path/to/episode.npz --model model.onnx

Runtime guarantees:
    * 10 Hz loop paced with ``time.perf_counter`` — deadline misses are
      logged and the scheduler resyncs on the next tick.
    * Ctrl-C in live mode always sends a zero-throttle stop command before
      exiting so the robot coasts to a stop even if the Python interpreter
      is being torn down.

Dependencies are strictly inside ``runtime/requirements.txt``: no torch, no
ROS, no Docker.  The repo-root ``serial_controller.py`` is reached via a
``sys.path`` shim already handled inside :mod:`runtime.control`.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

# Repo-root shim so ``shared.*`` and ``serial_controller`` import cleanly even
# when this module is run as ``python -m runtime.drive`` from an arbitrary cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.dataset_format import read_shard  # noqa: E402 — after sys.path shim

from runtime.inference import Policy  # noqa: E402

logger = logging.getLogger("runtime.drive")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runtime.drive",
        description=(
            "CouchMo on-laptop inference loop.  'live' reads stereo cameras "
            "and drives the robot over serial; 'replay' scores a recorded "
            "episode shard against the same policy — no serial, no cameras."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=("live", "replay"),
        required=True,
        help="Input surface: real cameras+serial ('live') or recorded shard ('replay').",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the .onnx model exported by training.imitation.export_onnx.",
    )
    # Live-only args.
    parser.add_argument(
        "--left-cam",
        type=int,
        default=None,
        help="[live] OpenCV camera index for the left eye.",
    )
    parser.add_argument(
        "--right-cam",
        type=int,
        default=None,
        help="[live] OpenCV camera index for the right eye.",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="[live] Serial device string (COM3, /dev/tty.usbserial-XXXX, /dev/ttyUSB0).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="[live] Control loop rate in Hz.",
    )
    # Replay-only arg.
    parser.add_argument(
        "--episode",
        type=Path,
        default=None,
        help="[replay] Path to an episode .npz shard written by shared.dataset_format.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level name.",
    )
    return parser


# ---------------------------------------------------------------------------
# Replay mode
# ---------------------------------------------------------------------------

def _run_replay(args: argparse.Namespace) -> int:
    """Score an episode shard against the policy, printing side-by-side actions."""
    if args.episode is None:
        print(
            "error: --episode is required in replay mode", file=sys.stderr
        )
        return 2

    shard_path = args.episode
    if not shard_path.exists():
        print(f"error: episode shard not found: {shard_path}", file=sys.stderr)
        return 2

    shard = read_shard(shard_path)
    left = shard["left"]
    right = shard["right"]
    steer_rec = shard["steer"]
    throttle_rec = shard["throttle"]

    n = len(left)
    if n == 0:
        print("error: shard has zero samples", file=sys.stderr)
        return 2

    logger.info(
        "replay: shard=%s  n=%d  model=%s", shard_path, n, args.model
    )

    policy = Policy(args.model)
    # Import preprocess locally so module import stays minimal.
    from shared.preprocess import preprocess_pair  # noqa: PLC0415

    for i in range(n):
        pair = preprocess_pair(left[i], right[i])
        ps, pt = policy.predict_from_pair(pair)
        rs = float(steer_rec[i])
        rt = float(throttle_rec[i])
        print(
            f"step {i:3d}  predicted=({ps:+.3f},{pt:.3f})  "
            f"recorded=({rs:+.3f},{rt:.3f})"
        )

    return 0


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

def _run_live(args: argparse.Namespace) -> int:
    """Real-time 10 Hz loop: cameras → policy → serial."""
    missing = []
    if args.left_cam is None:
        missing.append("--left-cam")
    if args.right_cam is None:
        missing.append("--right-cam")
    if args.port is None:
        missing.append("--port")
    if missing:
        print(
            f"error: live mode requires {', '.join(missing)}", file=sys.stderr
        )
        return 2
    if args.rate <= 0:
        print(f"error: --rate must be > 0 (got {args.rate})", file=sys.stderr)
        return 2

    # Imports done here (not at module top) so test code that only exercises
    # replay mode doesn't pay for camera/serial imports.
    from runtime.camera import DualCamera  # noqa: PLC0415
    from runtime.control import Controller  # noqa: PLC0415

    # Graceful shutdown flag — flipped by SIGINT handler.  We intentionally
    # swallow the default KeyboardInterrupt propagation so the finally block
    # below can guarantee a zero-throttle final command.
    stop_requested = {"flag": False}

    def _on_sigint(_signum, _frame):
        if not stop_requested["flag"]:
            logger.info("SIGINT received — stopping after current tick")
        stop_requested["flag"] = True

    previous_handler = signal.signal(signal.SIGINT, _on_sigint)

    policy = Policy(args.model)
    period = 1.0 / float(args.rate)

    logger.info(
        "live: cams=(%d,%d)  port=%s  model=%s  rate=%.1f Hz",
        args.left_cam, args.right_cam, args.port, args.model, args.rate,
    )

    try:
        with DualCamera(left_index=args.left_cam, right_index=args.right_cam) as cams, \
                Controller(port=args.port) as ctl:
            next_tick = time.perf_counter()
            tick_count = 0
            while not stop_requested["flag"]:
                try:
                    left_bgr, right_bgr = cams.read()
                except RuntimeError as exc:
                    logger.error("camera read failed: %s — stopping", exc)
                    break

                steer, throttle = policy.predict(left_bgr, right_bgr)
                ok = ctl.send(steer=steer, throttle=throttle)
                if not ok:
                    logger.warning(
                        "serial send returned ERR/timeout  steer=%+0.3f throttle=%.3f",
                        steer, throttle,
                    )

                tick_count += 1
                next_tick += period
                sleep_for = next_tick - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    logger.warning(
                        "deadline miss: %.1f ms late (tick %d)",
                        -sleep_for * 1000.0, tick_count,
                    )
                    # Resync so we don't chase a receding deadline.
                    next_tick = time.perf_counter()

            # Fall through to finally — stop() runs there.
    finally:
        # Restore the previous signal handler before the function returns.
        signal.signal(signal.SIGINT, previous_handler)

    logger.info("live mode exited cleanly")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns the process exit code (0 on success)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.source == "replay":
        return _run_replay(args)
    if args.source == "live":
        return _run_live(args)
    # argparse guarantees we never reach this branch.
    print(f"error: unknown source {args.source!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
