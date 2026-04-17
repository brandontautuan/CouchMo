"""In-sim evaluation node for the CouchMo behavior cloning policy.

This module is the sim-side consumer of a trained BC policy. It subscribes
to the two camera topics, runs the stacked observation through an ONNX or
PyTorch checkpoint, and publishes ``geometry_msgs/Vector3`` on
``/policy/steer_throttle`` (``x=steer``, ``y=throttle``, ``z=0``) — the
same interface the waypoint expert publishes on (on ``/expert/steer_throttle``)
and that ``serial_controller.py`` consumes on the real robot. Positive
steer is RIGHT per the IRL protocol convention.

The module is split so it remains importable on bare hosts (no ROS):

* Model loading + preprocessing helpers — pure Python / NumPy / torch /
  onnxruntime, no ROS imports. All heavy imports are lazy so ``--help``
  works on a minimal host.
* :class:`EvalInSimNode` — thin ``rclpy.node.Node`` wrapper, defined only
  when ``rclpy`` is present.

The ``rclpy`` imports are gated with ``try/except ImportError`` so this
module imports cleanly on non-ROS hosts (developer laptops, CI). If
invoked without ROS, :func:`main` prints a clear message and exits 0
(not 1) so test harnesses / CLI checks don't mistake "no ROS installed"
for a failure.

Observation pipeline (must match the recorder's):
    left_bgr, right_bgr  →  shared.preprocess.preprocess_pair  →  (2, 84, 84) uint8
    push through FrameStacker(num_frames=4)                    →  (8, 84, 84) float32
    add batch axis                                             →  (1, 8, 84, 84) float32
    model forward                                              →  (1, 2) float32 = (steer, throttle)

ATS is built over the two ``Image`` topics only (``Vector3`` has no
``Header`` and cannot participate in ATS). Publish rate is driven by
incoming synced camera pairs — the URDF runs cameras at 10 Hz, which
matches the action protocol's 10 Hz contract.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from shared.preprocess import FrameStacker, preprocess_pair


# ---------------------------------------------------------------------------
# Model loading — lazy imports so `--help` works on a torch-only host.
# ---------------------------------------------------------------------------

InferFn = Callable[[np.ndarray], np.ndarray]
"""A model wrapper: takes (1, 8, 84, 84) float32, returns (1, 2) float32."""


def _load_onnx(model_path: Path) -> InferFn:
    """Load an ONNX model and return a callable that runs inference."""
    import onnxruntime as ort  # noqa: PLC0415  — lazy import

    sess = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    # Export uses input_names=["input"] (see training.imitation.export_onnx).
    input_name = sess.get_inputs()[0].name

    def infer(x: np.ndarray) -> np.ndarray:
        return sess.run(None, {input_name: x})[0]

    return infer


def _load_pt(model_path: Path) -> InferFn:
    """Load a Task-8a .pt checkpoint and return a callable that runs inference."""
    import torch  # noqa: PLC0415 — lazy import

    from training.imitation.model import BCPolicy  # noqa: PLC0415

    ckpt = torch.load(str(model_path), weights_only=True)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "arch" not in ckpt:
        raise ValueError(
            "Checkpoint format not recognized; expected Task-8a format "
            "{'state_dict': ..., 'arch': {'in_channels': int}}. "
            f"Got type={type(ckpt).__name__}, "
            f"keys={list(ckpt.keys()) if isinstance(ckpt, dict) else 'N/A'}"
        )

    model = BCPolicy(**ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def infer(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            y = model(torch.from_numpy(x))
        return y.numpy()

    return infer


def load_model(model_path: Path) -> InferFn:
    """Dispatch on file suffix (.onnx vs .pt) and return an inference callable.

    Raises:
        FileNotFoundError: if ``model_path`` does not exist.
        ValueError: if the suffix is not ``.onnx`` or ``.pt``.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return _load_onnx(model_path)
    if suffix == ".pt":
        return _load_pt(model_path)
    raise ValueError(
        f"Unsupported model suffix '{suffix}'; expected .onnx or .pt"
    )


# ---------------------------------------------------------------------------
# ROS 2 wrapper — gated so the module stays importable on bare hosts.
# ---------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node
    import message_filters
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Vector3

    ROS_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only off-ROS hosts
    ROS_AVAILABLE = False
    Node = object  # type: ignore[misc,assignment]


if ROS_AVAILABLE:

    class EvalInSimNode(Node):  # type: ignore[misc,valid-type]
        """ROS 2 node that runs a BC policy on synced stereo frames.

        Parameters (declared on the node; CLI ``--model`` / ``--*-topic`` /
        ``--rate-hz`` flags override them by passing them through
        ``parameter_overrides``):

        ``model_path`` (str, **required**)
            Path to ``.onnx`` (preferred) or ``.pt`` checkpoint.
        ``left_topic`` (str)
            Left camera image topic (default ``/left_cam/image_raw``).
        ``right_topic`` (str)
            Right camera image topic (default ``/right_cam/image_raw``).
        ``output_topic`` (str)
            Policy action topic (default ``/policy/steer_throttle``).
        ``sync_slop_s`` (float)
            ATS slop tolerance in seconds (default 0.05).
        ``publish_rate_hz`` (float)
            Informational only — effective publish rate is driven by synced
            camera arrivals (default 10.0).
        """

        def __init__(
            self, parameter_overrides: "list[rclpy.parameter.Parameter] | None" = None
        ) -> None:
            # ``parameter_overrides`` are applied BEFORE ``declare_parameter``
            # fires, so CLI-provided values are the values ``get_parameter``
            # returns inside ``__init__``. This is how we let ``--model`` etc.
            # flow into the node without re-running init.
            super().__init__(
                "eval_in_sim",
                parameter_overrides=parameter_overrides or [],
            )

            self.declare_parameter("model_path", "")
            self.declare_parameter("left_topic", "/left_cam/image_raw")
            self.declare_parameter("right_topic", "/right_cam/image_raw")
            self.declare_parameter("output_topic", "/policy/steer_throttle")
            self.declare_parameter("sync_slop_s", 0.05)
            self.declare_parameter("publish_rate_hz", 10.0)

            model_path_str = str(self.get_parameter("model_path").value or "")
            if not model_path_str:
                self.get_logger().error(
                    "eval_in_sim: 'model_path' parameter is required"
                )
                raise RuntimeError("model_path parameter is empty")
            model_path = Path(model_path_str)

            left_topic = str(self.get_parameter("left_topic").value)
            right_topic = str(self.get_parameter("right_topic").value)
            output_topic = str(self.get_parameter("output_topic").value)
            sync_slop = float(self.get_parameter("sync_slop_s").value)
            rate_hz = float(self.get_parameter("publish_rate_hz").value)

            # Load the model eagerly so we fail fast if the file is missing
            # or the suffix is unsupported (rather than on the first frame).
            self._infer: InferFn = load_model(model_path)

            # Inference-time observation state — one stacker per run.
            self._stacker = FrameStacker(num_frames=4)
            self._bridge = CvBridge()

            # ATS over the two Image topics only — Vector3 has no Header so
            # it cannot participate in ApproximateTimeSynchronizer.
            left_sub = message_filters.Subscriber(self, Image, left_topic)
            right_sub = message_filters.Subscriber(self, Image, right_topic)
            self._ats = message_filters.ApproximateTimeSynchronizer(
                [left_sub, right_sub],
                queue_size=10,
                slop=sync_slop,
            )
            self._ats.registerCallback(self._on_images)

            self._pub = self.create_publisher(Vector3, output_topic, 10)

            self.get_logger().info(
                f"eval_in_sim: model='{model_path}', "
                f"left='{left_topic}', right='{right_topic}', "
                f"output='{output_topic}', sync_slop={sync_slop}s, "
                f"publish_rate_hz={rate_hz:.1f} (driven by camera arrivals)"
            )

        def _on_images(self, left_msg: "Image", right_msg: "Image") -> None:  # noqa: F821
            """ATS callback: preprocess, infer, publish."""
            try:
                left_bgr: np.ndarray = self._bridge.imgmsg_to_cv2(
                    left_msg, desired_encoding="bgr8"
                )
                right_bgr: np.ndarray = self._bridge.imgmsg_to_cv2(
                    right_msg, desired_encoding="bgr8"
                )
            except Exception as exc:
                self.get_logger().error(
                    f"eval_in_sim: cv_bridge conversion failed: {exc!r}"
                )
                return

            try:
                pair = preprocess_pair(left_bgr, right_bgr)
                obs = self._stacker.push(pair)  # (8, 84, 84) float32
                x = obs[np.newaxis, :, :, :]    # (1, 8, 84, 84) float32
                y = self._infer(x)              # (1, 2) float32
            except Exception as exc:
                # Inference failure must not take down the callback silently.
                # Log and publish a safe zero-throttle hold so downstream
                # nodes see a deterministic value.
                self.get_logger().error(
                    f"eval_in_sim: inference raised: {exc!r} — "
                    "publishing zero throttle"
                )
                safe = Vector3()
                safe.x = 0.0
                safe.y = 0.0
                safe.z = 0.0
                self._pub.publish(safe)
                return

            steer = float(y[0, 0])
            throttle = float(y[0, 1])

            msg = Vector3()
            msg.x = steer
            msg.y = throttle
            msg.z = 0.0
            self._pub.publish(msg)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser. Safe to call without ROS installed."""
    parser = argparse.ArgumentParser(
        description=(
            "In-sim evaluation node: loads a trained BC policy (.onnx or .pt), "
            "subscribes to the stereo camera topics, and publishes Vector3 "
            "(x=steer, y=throttle, z=0) on /policy/steer_throttle at 10 Hz. "
            "Safe to import without ROS (--help works)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        dest="model_path",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the policy file (.onnx preferred, .pt also supported).",
    )
    parser.add_argument(
        "--left-topic",
        type=str,
        default="/left_cam/image_raw",
        help="Left camera Image topic.",
    )
    parser.add_argument(
        "--right-topic",
        type=str,
        default="/right_cam/image_raw",
        help="Right camera Image topic.",
    )
    parser.add_argument(
        "--output-topic",
        type=str,
        default="/policy/steer_throttle",
        help="Vector3 publication topic (x=steer, y=throttle, z=0).",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help=(
            "Informational publish rate; effective rate is driven by synced "
            "camera arrivals (cameras run at 10 Hz per the URDF)."
        ),
    )
    parser.add_argument(
        "--sync-slop-s",
        type=float,
        default=0.05,
        help="ApproximateTimeSynchronizer slop tolerance in seconds.",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    """Entry point. Safe to call without ROS — prints a message and exits 0."""
    parser = _build_arg_parser()
    ns = parser.parse_args(args)

    if not ROS_AVAILABLE:
        # Explicit: exit 0 (not 1). The plan is explicit that "no ROS" is
        # not a failure — it's the expected state on the dev laptop, so
        # `python -m training.imitation.eval_in_sim --help` and any CI
        # import-check should not see a non-zero exit.
        print(
            "rclpy not available; this script must run inside the sim container"
        )
        sys.exit(0)

    # ROS is available — boot the node. Forward the CLI flags as ROS 2
    # parameter overrides so the declared parameters see the user's values
    # at construction time.
    parameter_overrides = [
        rclpy.parameter.Parameter("model_path", value=str(ns.model_path)),
        rclpy.parameter.Parameter("left_topic", value=str(ns.left_topic)),
        rclpy.parameter.Parameter("right_topic", value=str(ns.right_topic)),
        rclpy.parameter.Parameter("output_topic", value=str(ns.output_topic)),
        rclpy.parameter.Parameter("publish_rate_hz", value=float(ns.rate_hz)),
        rclpy.parameter.Parameter("sync_slop_s", value=float(ns.sync_slop_s)),
    ]

    rclpy.init(args=args)
    try:
        node = EvalInSimNode(parameter_overrides=parameter_overrides)  # type: ignore[misc]
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
