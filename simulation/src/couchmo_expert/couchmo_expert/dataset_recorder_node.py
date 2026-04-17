"""Dataset recorder node — writes shared ``.npz`` shards to a bind-mounted volume.

This module is the sim-side writer of the imitation-learning dataset.  It
subscribes to the two camera topics and the expert action topic, aligns frames
across time, and flushes buffers to disk as ``.npz`` shards in the format
defined by ``shared.dataset_format``.

Architecture note on ``ApproximateTimeSynchronizer`` (ATS) vs. Vector3
-----------------------------------------------------------------------
The original plan specifies ATS over all three topics:

    /left_cam/image_raw
    /right_cam/image_raw
    /expert/steer_throttle

``message_filters.ApproximateTimeSynchronizer`` requires every topic to carry a
``std_msgs/Header`` so it can compare ROS timestamps.  ``sensor_msgs/Image``
has one; ``geometry_msgs/Vector3`` does **not**.  We therefore cannot pass the
Vector3 subscription directly to ATS.

Resolution used here (preserves the plan's intent while respecting
message_filters constraints):

* ATS is built over *only* the two ``Image`` topics
  (``slop`` = ``sync_slop_s`` param, default 0.05 s).
* A **plain subscription** to ``/expert/steer_throttle`` caches the latest
  ``(steer, throttle)`` pair and its receive-wall-clock time into
  ``self._latest_action``.
* In the synced image callback, if ``_latest_action is None`` (no expert
  message received yet) the sample is **dropped** — we never write a
  zero-throttle sample to avoid poisoning the dataset.
* Because the cameras run at 10 Hz (URDF ``update_rate=10``) and the waypoint
  expert publishes at 10 Hz (Task 5), effective alignment is ≈10 Hz naturally.

The module is split into two halves following the project pattern:

* :class:`ShardBuffer` — pure-Python / NumPy class, importable on any host
  without ROS.  This is what unit tests exercise.
* :class:`DatasetRecorderNode` — thin ``rclpy.node.Node`` wrapper, defined only
  when ``rclpy`` is present.

The ``rclpy`` (and associated ROS) imports are gated with ``try/except
ImportError`` so this module imports cleanly on non-ROS hosts.  :func:`main`
raises ``RuntimeError`` at runtime if invoked without ROS.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Pure-Python shard buffer logic (no ROS dependency)
# ---------------------------------------------------------------------------


class ShardBuffer:
    """Accumulate per-sample arrays and flush full shards to disk.

    The class is pure Python + NumPy (no ``rclpy`` / ROS imports) so it can be
    unit-tested on a bare developer host.

    Parameters
    ----------
    episode_dir:
        Directory where shards are written, e.g. ``/workspace/data/ep_abc/``.
        The directory is created on first flush if it does not exist.
    samples_per_shard:
        Number of samples before a flush is triggered automatically.  Partial
        shards (fewer samples than this) can be flushed explicitly via
        :meth:`flush`.
    """

    def __init__(self, episode_dir: Path | str, *, samples_per_shard: int = 200) -> None:
        if samples_per_shard < 1:
            raise ValueError(
                f"samples_per_shard must be >= 1; got {samples_per_shard}"
            )
        self._episode_dir = Path(episode_dir)
        self._samples_per_shard = int(samples_per_shard)
        self._shard_index: int = 0
        self._total_samples: int = 0
        self._clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clear(self) -> None:
        """Reset the in-flight per-key lists."""
        self._buf: dict[str, list[Any]] = {
            "left": [],
            "right": [],
            "steer": [],
            "throttle": [],
            "t": [],
        }

    def _flush_internal(self) -> Path | None:
        """Stack lists → numpy, call ``write_shard``, bump counter.

        Returns the path of the shard file just written, or ``None`` if the
        buffer was empty.
        """
        n = len(self._buf["left"])
        if n == 0:
            return None

        # Lazy import so the class stays importable on bare hosts without numpy
        # being installed — in practice numpy is always available, but the gate
        # mirrors the project pattern of keeping pure logic dependency-free.
        from shared.dataset_format import write_shard  # noqa: PLC0415

        left = np.stack(self._buf["left"], axis=0).astype(np.uint8)
        right = np.stack(self._buf["right"], axis=0).astype(np.uint8)
        steer = np.array(self._buf["steer"], dtype=np.float32)
        throttle = np.array(self._buf["throttle"], dtype=np.float32)
        t = np.array(self._buf["t"], dtype=np.float64)

        shard_path = self._episode_dir / f"shard_{self._shard_index}.npz"
        write_shard(shard_path, left=left, right=right, steer=steer, throttle=throttle, t=t)

        self._shard_index += 1
        self._total_samples += n
        self._clear()
        return shard_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_samples(self) -> int:
        """Total number of samples successfully written to completed shards."""
        return self._total_samples

    @property
    def buffered_samples(self) -> int:
        """Number of samples currently in the in-flight buffer (not yet flushed)."""
        return len(self._buf["left"])

    @property
    def shard_index(self) -> int:
        """Index of the next shard that will be written."""
        return self._shard_index

    def add(
        self,
        *,
        left: np.ndarray,
        right: np.ndarray,
        steer: float,
        throttle: float,
        t: float,
    ) -> Path | None:
        """Append one sample to the buffer.

        Parameters
        ----------
        left:
            ``(H, W, 3)`` uint8 BGR frame from the left camera.
        right:
            ``(H, W, 3)`` uint8 BGR frame from the right camera.
        steer:
            Steering command in ``[-1, 1]``.
        throttle:
            Throttle command in ``[0, 1]``.
        t:
            ``t`` is recorded as seconds elapsed since episode start, derived from ROS
            ``get_clock()``. When ``use_sim_time=True`` (the default when launched
            alongside Gazebo) this follows ``/clock``, so the timeline freezes if the
            simulator is paused — duplicate timestamps in a shard indicate a paused sim,
            not a recorder bug.

        Returns
        -------
        Path | None
            The path of the shard just written if a flush occurred, else
            ``None``.
        """
        self._buf["left"].append(left)
        self._buf["right"].append(right)
        self._buf["steer"].append(float(steer))
        self._buf["throttle"].append(float(throttle))
        self._buf["t"].append(float(t))

        if len(self._buf["left"]) >= self._samples_per_shard:
            return self._flush_internal()
        return None

    def flush(self) -> Path | None:
        """Flush the partial buffer to disk immediately.

        Call this at node shutdown to persist any samples that did not fill a
        complete shard.

        Returns
        -------
        Path | None
            The path of the shard written, or ``None`` if the buffer was empty.
        """
        return self._flush_internal()


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

    class DatasetRecorderNode(Node):  # type: ignore[misc,valid-type]
        """ROS 2 node that records synchronised camera frames + expert actions.

        Parameters (declared on the node; set at launch or via YAML):

        ``episode_id`` (str, **required**)
            Unique identifier for this recording run, e.g.
            ``"ep_20260416_181234"``.  The node refuses to start without it so
            two runs can never silently collide on the same directory.
        ``output_root`` (str)
            Root directory for all episodes (default ``/workspace/data``).
            The shard files land at
            ``<output_root>/<episode_id>/shard_<n>.npz``.
        ``world`` (str)
            Gazebo world name, recorded in the manifest (default ``""``).
        ``notes`` (str)
            Free-text notes, recorded in the manifest (default ``""``).
        ``samples_per_shard`` (int)
            Flush a new shard every N samples (default 200).
        ``left_topic`` (str)
            Left camera image topic (default ``/left_cam/image_raw``).
        ``right_topic`` (str)
            Right camera image topic (default ``/right_cam/image_raw``).
        ``steer_throttle_topic`` (str)
            Expert action topic (default ``/expert/steer_throttle``).
        ``sample_rate_hz`` (float)
            Informational only — the actual rate is driven by synced image
            arrivals, which happen at 10 Hz when the URDF ``update_rate=10``
            (default 10.0).
        ``sync_slop_s`` (float)
            ATS slop tolerance in seconds (default 0.05).
        """

        def __init__(self) -> None:
            super().__init__("dataset_recorder")

            # Declare parameters
            self.declare_parameter("episode_id", "")
            self.declare_parameter("output_root", "/workspace/data")
            self.declare_parameter("world", "")
            self.declare_parameter("notes", "")
            self.declare_parameter("samples_per_shard", 200)
            self.declare_parameter("left_topic", "/left_cam/image_raw")
            self.declare_parameter("right_topic", "/right_cam/image_raw")
            self.declare_parameter("steer_throttle_topic", "/expert/steer_throttle")
            self.declare_parameter("sample_rate_hz", 10.0)
            self.declare_parameter("sync_slop_s", 0.05)

            # Validate required episode_id
            episode_id = str(self.get_parameter("episode_id").value or "")
            if not episode_id:
                self.get_logger().error(
                    "dataset_recorder: 'episode_id' parameter is required — "
                    "set it explicitly so runs never collide silently"
                )
                raise RuntimeError("episode_id parameter is empty")

            output_root = Path(str(self.get_parameter("output_root").value))
            self._world = str(self.get_parameter("world").value or "")
            self._notes = str(self.get_parameter("notes").value or "")
            samples_per_shard = int(self.get_parameter("samples_per_shard").value)
            left_topic = str(self.get_parameter("left_topic").value)
            right_topic = str(self.get_parameter("right_topic").value)
            steer_throttle_topic = str(self.get_parameter("steer_throttle_topic").value)
            sync_slop = float(self.get_parameter("sync_slop_s").value)

            # Build episode directory
            episode_dir = output_root / episode_id
            self._episode_id = episode_id
            self._output_root = output_root
            self._manifest_path = output_root / "manifest.json"

            # Create shard buffer
            self._buffer = ShardBuffer(episode_dir, samples_per_shard=samples_per_shard)

            # cv_bridge for image conversion
            self._bridge = CvBridge()

            # Episode timing
            self._start_sec: float | None = None
            self._start_iso: str = ""

            # Latest cached action from the plain steer_throttle subscription.
            # None until the first message is received.
            self._latest_action: tuple[float, float] | None = None

            # Throttled WARN: tracks the last time we warned about missing action.
            # None means we have not yet emitted any warning.
            self._last_noaction_warn_ns: int | None = None

            # Build ATS over the two Image topics only — Vector3 has no Header
            # so it cannot participate in ApproximateTimeSynchronizer.
            left_sub = message_filters.Subscriber(self, Image, left_topic)
            right_sub = message_filters.Subscriber(self, Image, right_topic)
            self._ats = message_filters.ApproximateTimeSynchronizer(
                [left_sub, right_sub],
                queue_size=10,
                slop=sync_slop,
            )
            self._ats.registerCallback(self._on_images)

            # Plain subscription for the Vector3 action — caches latest only
            self._action_sub = self.create_subscription(
                Vector3,
                steer_throttle_topic,
                self._on_action,
                10,
            )

            self.get_logger().info(
                f"dataset_recorder: episode_id='{episode_id}', "
                f"output_root='{output_root}', "
                f"episode_dir='{episode_dir}', "
                f"samples_per_shard={samples_per_shard}, "
                f"sync_slop={sync_slop}s, "
                f"left='{left_topic}', right='{right_topic}', "
                f"action='{steer_throttle_topic}'"
            )

        # ------------------------------------------------------------------
        # Callbacks
        # ------------------------------------------------------------------

        def _on_action(self, msg: "Vector3") -> None:  # noqa: F821
            """Cache the latest steer (msg.x) and throttle (msg.y)."""
            self._latest_action = (float(msg.x), float(msg.y))

        def _on_images(self, left_msg: "Image", right_msg: "Image") -> None:  # noqa: F821
            """ATS callback: record one aligned sample if an action is cached."""
            if self._latest_action is None:
                # No expert action yet — drop this frame pair to avoid
                # zero-throttle samples polluting the dataset.
                now_ns = self.get_clock().now().nanoseconds
                if (
                    self._last_noaction_warn_ns is None
                    or (now_ns - self._last_noaction_warn_ns) >= 5_000_000_000
                ):
                    self.get_logger().warn(
                        "dataset_recorder: no /expert/steer_throttle received yet"
                        " — dropping synced image pairs (check the waypoint_expert node)"
                    )
                    self._last_noaction_warn_ns = now_ns
                return

            # Record episode start on first accepted sample
            if self._start_sec is None:
                self._start_sec = self.get_clock().now().nanoseconds * 1e-9
                self._start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                self.get_logger().info(
                    f"dataset_recorder: episode start at {self._start_iso}"
                )

            # Convert ROS Image messages to numpy BGR arrays via cv_bridge
            try:
                left_bgr: np.ndarray = self._bridge.imgmsg_to_cv2(
                    left_msg, desired_encoding="bgr8"
                )
                right_bgr: np.ndarray = self._bridge.imgmsg_to_cv2(
                    right_msg, desired_encoding="bgr8"
                )
            except Exception as exc:
                self.get_logger().error(
                    f"dataset_recorder: cv_bridge conversion failed: {exc!r}"
                )
                return

            steer, throttle = self._latest_action
            t = self.get_clock().now().nanoseconds * 1e-9 - self._start_sec

            shard_path = self._buffer.add(
                left=left_bgr,
                right=right_bgr,
                steer=steer,
                throttle=throttle,
                t=t,
            )
            if shard_path is not None:
                self.get_logger().info(
                    f"dataset_recorder: flushed shard → {shard_path} "
                    f"(total {self._buffer.total_samples} samples)"
                )

        # ------------------------------------------------------------------
        # Lifecycle
        # ------------------------------------------------------------------

        def flush(self) -> None:
            """Flush the partial buffer and write the manifest entry.

            Call this from ``main()``'s ``finally`` block before
            ``destroy_node()`` to ensure no samples are lost.
            """
            shard_path = self._buffer.flush()
            if shard_path is not None:
                self.get_logger().info(
                    f"dataset_recorder: flushed partial shard → {shard_path}"
                )

            total = self._buffer.total_samples
            if total == 0:
                self.get_logger().warn(
                    "dataset_recorder: no samples recorded — skipping manifest update"
                )
                return

            # Load or create manifest, then append this episode's entry.
            from shared.dataset_format import EpisodeMeta, Manifest  # noqa: PLC0415

            if self._manifest_path.is_file():
                manifest = Manifest.from_json(self._manifest_path)
            else:
                manifest = Manifest()

            manifest.episodes.append(
                EpisodeMeta(
                    id=self._episode_id,
                    n_samples=total,
                    created_utc=self._start_iso or datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    # Per plan: put the episode *directory* name in shard_path
                    # so the training loader can glob all shards inside it.
                    shard_path=self._episode_id,
                    world=self._world,
                    notes=self._notes,
                )
            )
            manifest.to_json(self._manifest_path)
            self.get_logger().info(
                f"dataset_recorder: manifest updated → {self._manifest_path} "
                f"({total} samples, episode '{self._episode_id}')"
            )


def main(args: list[str] | None = None) -> None:
    """Entry point for the ``dataset_recorder_node`` console script."""
    if not ROS_AVAILABLE:
        raise RuntimeError(
            "rclpy not available; this node must run inside the sim container"
        )
    rclpy.init(args=args)
    node = DatasetRecorderNode()
    try:
        rclpy.spin(node)
    finally:
        node.flush()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
