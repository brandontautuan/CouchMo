"""Pure-pursuit waypoint expert for the CouchMo campus simulation.

This module is the data-generation oracle used by the imitation-learning
pipeline. It tracks a pre-recorded sidewalk centerline (CSV produced by
``simulation/scripts/kml_to_waypoints.py``) and emits
``(steer, throttle)`` actions at a fixed 10 Hz rate. The emitted
``(steer, throttle)`` tuple is deliberately identical to the IRL serial
protocol consumed by ``serial_controller.py`` so the expert's actions
and the eventually-learned policy's actions live on the same interface.

The file is split into two halves:

* :class:`PurePursuitExpert` — pure-Python / NumPy class, importable on
  any host (Windows/macOS/Linux) without ROS. This is what the unit
  tests exercise.
* :class:`WaypointExpertNode` — thin :class:`rclpy.node.Node` wrapper
  that subscribes odometry, runs the pure class at
  ``publish_rate_hz`` via a timer, and publishes
  ``geometry_msgs/Vector3`` (``x=steer``, ``y=throttle``, ``z=0``).

The ``rclpy`` imports are gated with ``try/except ImportError`` so this
module imports cleanly on non-ROS hosts; the ROS wrapper is only
defined when ``rclpy`` is present. :func:`main` raises at runtime if
invoked without ROS.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# IRL serial protocol sign convention.
#
# Pinned from ``serial_controller.py``:
#   - line 67:   ``steer:    -1.0 (full left) to 1.0 (full right)``
#   - line 123:  ``("turn left",  -0.5, 0.3),``
#   - line 124:  ``("turn right",  0.5, 0.3),``
#
# Under the conventional right-handed robot frame (x forward, y left, z up,
# yaw positive CCW) the pure-pursuit curvature
#
#     κ = 2 · y_robot / L²
#
# is positive when the target waypoint lies to the LEFT of the robot
# (``y_robot > 0``), which in a well-behaved kinematic model means "turn
# left". But the IRL protocol defines positive steer as RIGHT. To bridge
# the two conventions we negate: positive curvature (target-on-left) →
# negative steer (full-left) as the IRL protocol expects.
STEER_POS_DIR: float = -1.0
# ---------------------------------------------------------------------------


class PurePursuitExpert:
    """Waypoint-tracking expert using the standard pure-pursuit law.

    The class is pure Python + NumPy (no ``rclpy`` / ROS imports) so it
    can be unit-tested on a bare developer host. Construct it once with
    the campus waypoint polyline, then call :meth:`compute` each control
    tick to produce a ``(steer, throttle, info)`` triple.

    Parameters
    ----------
    waypoints:
        ``(N, 2)`` array of ``(x, y)`` metres, in path order. Must have
        ``N >= 2``. The CSV produced by ``kml_to_waypoints.py`` already
        satisfies the monotone-arc-length requirement this class
        assumes (consecutive duplicates dropped).
    lookahead_min_m:
        Lookahead distance used at zero speed (metres).
    lookahead_max_m:
        Lookahead distance at / above ``lookahead_speed_cap_mps``. Must
        satisfy ``lookahead_max_m >= lookahead_min_m``.
    lookahead_speed_cap_mps:
        Speed (m/s) at which the lookahead saturates to
        ``lookahead_max_m``. Between 0 and this, the lookahead scales
        linearly.
    max_throttle:
        Upper cap on the emitted throttle command, in ``[0, 1]``.
    curvature_throttle_gain:
        Slows the robot on curvy parts of the path:
        ``throttle = max_throttle * (1 - gain * |κ|)`` (then clipped to
        ``[0, max_throttle]``).
    wheelbase_m:
        Informational for CouchMo's differential-drive kinematics — the
        wheelbase does NOT enter the curvature-to-steer mapping on a
        diff-drive platform (a diff drive can match any curvature at
        zero translational speed). Retained here for parameter-surface
        parity with future Ackermann extensions and for logging.
    max_curvature_at_full_steer:
        Curvature magnitude (1/m) that corresponds to a full left/right
        steer command. The steer output is
        ``κ / max_curvature_at_full_steer * STEER_POS_DIR``, clipped to
        ``[-1, 1]``.
    """

    def __init__(
        self,
        *,
        waypoints: np.ndarray,
        lookahead_min_m: float,
        lookahead_max_m: float,
        lookahead_speed_cap_mps: float,
        max_throttle: float,
        curvature_throttle_gain: float,
        wheelbase_m: float,
        max_curvature_at_full_steer: float,
    ) -> None:
        wps = np.asarray(waypoints, dtype=np.float64)
        if wps.ndim != 2 or wps.shape[1] != 2:
            raise ValueError(
                f"waypoints must be (N, 2); got shape {wps.shape}"
            )
        if wps.shape[0] < 2:
            raise ValueError(
                f"waypoints must contain at least 2 rows; got {wps.shape[0]}"
            )
        if lookahead_min_m <= 0.0 or lookahead_max_m < lookahead_min_m:
            raise ValueError(
                f"require 0 < lookahead_min_m <= lookahead_max_m; "
                f"got min={lookahead_min_m}, max={lookahead_max_m}"
            )
        if lookahead_speed_cap_mps <= 0.0:
            raise ValueError(
                f"lookahead_speed_cap_mps must be > 0; got {lookahead_speed_cap_mps}"
            )
        if not 0.0 <= max_throttle <= 1.0:
            raise ValueError(
                f"max_throttle must be in [0, 1]; got {max_throttle}"
            )
        if max_curvature_at_full_steer <= 0.0:
            raise ValueError(
                f"max_curvature_at_full_steer must be > 0; "
                f"got {max_curvature_at_full_steer}"
            )

        self._waypoints = wps
        seg = np.diff(wps, axis=0)
        self._segment_lengths = np.hypot(seg[:, 0], seg[:, 1])
        self._lookahead_min_m = float(lookahead_min_m)
        self._lookahead_max_m = float(lookahead_max_m)
        self._lookahead_speed_cap_mps = float(lookahead_speed_cap_mps)
        self._max_throttle = float(max_throttle)
        self._curvature_throttle_gain = float(curvature_throttle_gain)
        self._wheelbase_m = float(wheelbase_m)
        self._max_curvature_at_full_steer = float(max_curvature_at_full_steer)

    @property
    def num_waypoints(self) -> int:
        """Return the number of stored waypoints."""
        return int(self._waypoints.shape[0])

    def _lookahead_for_speed(self, speed_mps: float) -> float:
        """Return the linearly-scheduled lookahead distance in metres.

        Clamped at both ends so negative speeds use ``lookahead_min_m``
        and speeds above the cap saturate at ``lookahead_max_m``.
        """
        s = max(0.0, float(speed_mps))
        if s >= self._lookahead_speed_cap_mps:
            return self._lookahead_max_m
        frac = s / self._lookahead_speed_cap_mps
        return (
            self._lookahead_min_m
            + (self._lookahead_max_m - self._lookahead_min_m) * frac
        )

    def _closest_waypoint_index(self, pose_xy: tuple[float, float]) -> int:
        """Return the index of the waypoint nearest ``pose_xy``."""
        dx = self._waypoints[:, 0] - pose_xy[0]
        dy = self._waypoints[:, 1] - pose_xy[1]
        return int(np.argmin(dx * dx + dy * dy))

    def _find_target_index(self, closest_idx: int, lookahead_m: float) -> int:
        """Walk forward along the polyline from ``closest_idx``.

        Returns the first index whose cumulative arc length from
        ``closest_idx`` meets or exceeds ``lookahead_m``, clamped to the
        last waypoint if the path ends before the lookahead is reached.
        """
        cumulative = 0.0
        i = int(closest_idx)
        n = self._waypoints.shape[0]
        while i + 1 < n and cumulative < lookahead_m:
            cumulative += float(self._segment_lengths[i])
            i += 1
        return i

    def _is_past_last(self, pose_xy: tuple[float, float]) -> bool:
        """Return True if ``pose_xy`` projects past the final waypoint.

        Projection is along the last segment's direction; a positive
        projection past the last waypoint means the robot has driven
        off the end of the path.
        """
        n = self._waypoints.shape[0]
        last = self._waypoints[n - 1]
        prev = self._waypoints[n - 2]
        seg = last - prev
        seg_norm = float(math.hypot(seg[0], seg[1]))
        if seg_norm == 0.0:
            return False
        disp_x = pose_xy[0] - last[0]
        disp_y = pose_xy[1] - last[1]
        projection = (disp_x * seg[0] + disp_y * seg[1]) / seg_norm
        return projection > 0.0

    def compute(
        self,
        *,
        pose_xy: tuple[float, float],
        yaw_rad: float,
        speed_mps: float,
    ) -> tuple[float, float, dict[str, Any]]:
        """Produce one ``(steer, throttle)`` command.

        Parameters
        ----------
        pose_xy:
            Current robot position in world metres (ENU-like, matching
            the waypoint CSV's frame).
        yaw_rad:
            Current heading (radians, CCW from +x axis).
        speed_mps:
            Current forward speed (m/s). Used to schedule the lookahead.

        Returns
        -------
        tuple[float, float, dict[str, Any]]
            ``(steer, throttle, info)`` where ``steer ∈ [-1, 1]``,
            ``throttle ∈ [0, max_throttle]``, and ``info`` always
            contains the keys ``target_idx``, ``lookahead_m``,
            ``lateral_error_m`` and ``done``.
        """
        closest_idx = self._closest_waypoint_index(pose_xy)
        lookahead_m = self._lookahead_for_speed(speed_mps)
        target_idx = self._find_target_index(closest_idx, lookahead_m)
        target = self._waypoints[target_idx]

        dx = float(target[0] - pose_xy[0])
        dy = float(target[1] - pose_xy[1])
        cos_psi = math.cos(yaw_rad)
        sin_psi = math.sin(yaw_rad)
        x_robot = cos_psi * dx + sin_psi * dy
        y_robot = -sin_psi * dx + cos_psi * dy
        target_dist = math.hypot(x_robot, y_robot)

        if target_dist < 1e-6:
            kappa = 0.0
        else:
            kappa = 2.0 * y_robot / (target_dist * target_dist)

        steer_raw = STEER_POS_DIR * kappa / self._max_curvature_at_full_steer
        steer = max(-1.0, min(1.0, steer_raw))

        throttle_raw = self._max_throttle * (
            1.0 - self._curvature_throttle_gain * abs(kappa)
        )
        throttle = max(0.0, min(self._max_throttle, throttle_raw))

        closest_xy = self._waypoints[closest_idx]
        lateral_error_m = float(
            math.hypot(
                pose_xy[0] - float(closest_xy[0]),
                pose_xy[1] - float(closest_xy[1]),
            )
        )

        done = False
        n = self._waypoints.shape[0]
        if closest_idx == n - 1 and self._is_past_last(pose_xy):
            throttle = 0.0
            done = True

        info: dict[str, Any] = {
            "target_idx": int(target_idx),
            "lookahead_m": float(lookahead_m),
            "lateral_error_m": lateral_error_m,
            "done": bool(done),
        }
        return float(steer), float(throttle), info


def load_waypoints_csv(path: Path) -> np.ndarray:
    """Load ``(N, 2)`` ``(x, y)`` waypoints from an ``x_m,y_m,s_m`` CSV.

    Parameters
    ----------
    path:
        Filesystem path to a CSV with header row ``x_m,y_m,s_m`` (the
        format emitted by ``kml_to_waypoints.py``). The ``s_m`` column
        is ignored — per-segment arc length is recomputed from the
        Euclidean distance between consecutive ``(x, y)`` samples.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` float64 array of ``(x, y)`` metres.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the CSV has fewer than 2 valid rows.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"waypoints CSV not found: {csv_path}")
    rows: list[tuple[float, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append((float(row["x_m"]), float(row["y_m"])))
    if len(rows) < 2:
        raise ValueError(
            f"waypoints CSV {csv_path} must contain at least 2 rows; "
            f"got {len(rows)}"
        )
    return np.array(rows, dtype=np.float64)


# --- ROS 2 wrapper -----------------------------------------------------------
# Gated so the module stays importable on bare hosts (no ROS installed),
# which matters for running unit tests outside the sim container.
try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Vector3

    ROS_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only off-ROS hosts
    ROS_AVAILABLE = False
    Node = object  # type: ignore[misc,assignment]


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract the yaw (rad) component of an XYZW unit quaternion.

    Uses the standard ZYX Tait-Bryan formula
    (matches ``tf_transformations.euler_from_quaternion(q)[2]``).
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


if ROS_AVAILABLE:

    class WaypointExpertNode(Node):  # type: ignore[misc,valid-type]
        """ROS 2 wrapper that runs :class:`PurePursuitExpert` on a timer.

        Parameters (declared on the node so a YAML config file can set
        them; see ``config/waypoint_follower.yaml``):

        ``waypoints_csv`` (str, required)
            Absolute path to the ``x_m,y_m,s_m`` waypoint CSV.
        ``odom_topic`` (str)
            Odometry subscription topic (default ``/odom``).
        ``output_topic`` (str)
            ``geometry_msgs/Vector3`` publisher topic
            (default ``/expert/steer_throttle``).
        ``publish_rate_hz`` (float)
            Control-loop rate (default 10 Hz).
        ``lookahead_min_m``, ``lookahead_max_m``,
        ``lookahead_speed_cap_mps``, ``max_throttle``,
        ``curvature_throttle_gain``, ``wheelbase_m``,
        ``max_curvature_at_full_steer``
            Forwarded verbatim to :class:`PurePursuitExpert`.

        The node caches the latest odometry and only publishes once at
        least one odometry message has been received (avoids emitting a
        bogus "drive forward" on startup).
        """

        def __init__(self) -> None:
            super().__init__("waypoint_expert")
            self.declare_parameter("waypoints_csv", "")
            self.declare_parameter("odom_topic", "/odom")
            self.declare_parameter("output_topic", "/expert/steer_throttle")
            self.declare_parameter("publish_rate_hz", 10.0)
            self.declare_parameter("corridor_width_m", 3.0)
            self.declare_parameter("lookahead_min_m", 1.0)
            self.declare_parameter("lookahead_max_m", 4.0)
            self.declare_parameter("lookahead_speed_cap_mps", 2.0)
            self.declare_parameter("max_throttle", 0.6)
            self.declare_parameter("curvature_throttle_gain", 1.0)
            self.declare_parameter("wheelbase_m", 0.6)
            self.declare_parameter("max_curvature_at_full_steer", 1.0)

            csv_param = str(self.get_parameter("waypoints_csv").value or "")
            if not csv_param:
                self.get_logger().error(
                    "waypoint_expert: 'waypoints_csv' parameter is required"
                )
                raise RuntimeError("waypoints_csv parameter is empty")
            csv_path = Path(csv_param)
            if not csv_path.is_file():
                self.get_logger().error(
                    f"waypoint_expert: waypoints CSV not found at {csv_path}"
                )
                raise FileNotFoundError(csv_path)
            waypoints = load_waypoints_csv(csv_path)

            self._expert = PurePursuitExpert(
                waypoints=waypoints,
                lookahead_min_m=float(
                    self.get_parameter("lookahead_min_m").value
                ),
                lookahead_max_m=float(
                    self.get_parameter("lookahead_max_m").value
                ),
                lookahead_speed_cap_mps=float(
                    self.get_parameter("lookahead_speed_cap_mps").value
                ),
                max_throttle=float(self.get_parameter("max_throttle").value),
                curvature_throttle_gain=float(
                    self.get_parameter("curvature_throttle_gain").value
                ),
                wheelbase_m=float(self.get_parameter("wheelbase_m").value),
                max_curvature_at_full_steer=float(
                    self.get_parameter("max_curvature_at_full_steer").value
                ),
            )

            self._pose_xy: tuple[float, float] = (0.0, 0.0)
            self._yaw_rad: float = 0.0
            self._speed_mps: float = 0.0
            self._have_odom: bool = False

            odom_topic = str(self.get_parameter("odom_topic").value)
            output_topic = str(self.get_parameter("output_topic").value)
            rate_hz = float(self.get_parameter("publish_rate_hz").value)
            if rate_hz <= 0.0:
                raise ValueError(
                    f"publish_rate_hz must be > 0; got {rate_hz}"
                )

            self._sub = self.create_subscription(
                Odometry, odom_topic, self._on_odom, 10
            )
            self._pub = self.create_publisher(Vector3, output_topic, 10)
            self._timer = self.create_timer(1.0 / rate_hz, self._on_tick)

            self.get_logger().info(
                f"waypoint_expert: loaded {waypoints.shape[0]} waypoints "
                f"from {csv_path}; publishing at {rate_hz:.1f} Hz on "
                f"{output_topic}; subscribing {odom_topic}"
            )

        def _on_odom(self, msg: "Odometry") -> None:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            self._pose_xy = (float(p.x), float(p.y))
            self._yaw_rad = _yaw_from_quat(
                float(q.x), float(q.y), float(q.z), float(q.w)
            )
            self._speed_mps = float(msg.twist.twist.linear.x)
            self._have_odom = True

        def _on_tick(self) -> None:
            if not self._have_odom:
                return
            steer, throttle, info = self._expert.compute(
                pose_xy=self._pose_xy,
                yaw_rad=self._yaw_rad,
                speed_mps=self._speed_mps,
            )
            msg = Vector3()
            msg.x = float(steer)
            msg.y = float(throttle)
            msg.z = 0.0
            self._pub.publish(msg)
            if info["done"]:
                self.get_logger().info(
                    "waypoint_expert: path complete — emitting zero throttle"
                )


def main(args: list[str] | None = None) -> None:
    """Entry point for the ``waypoint_expert_node`` console script."""
    if not ROS_AVAILABLE:
        raise RuntimeError(
            "rclpy not available; this node must run inside the sim container"
        )
    rclpy.init(args=args)
    node = WaypointExpertNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
