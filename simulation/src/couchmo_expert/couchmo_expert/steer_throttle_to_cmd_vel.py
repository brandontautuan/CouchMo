"""Action adapter: maps ``(steer, throttle)`` to ``geometry_msgs/Twist``.

This module bridges the IRL serial-protocol action space (used by the
waypoint expert and the learned policy) to the ``/cmd_vel`` topic that
Gazebo's differential-drive plugin and the physical robot controller
both consume.

The file is split into two halves following the project pattern:

* :func:`steer_throttle_to_twist` — pure Python function, importable on
  any host (Windows/macOS/Linux) without ROS. This is what the unit
  tests exercise.
* :class:`SteerThrottleToCmdVelNode` — thin :class:`rclpy.node.Node`
  wrapper that subscribes ``/expert/steer_throttle`` (or
  ``/policy/steer_throttle``), caches the latest input, and re-publishes
  a ``geometry_msgs/Twist`` on ``/cmd_vel`` at a fixed 10 Hz rate via a
  timer. Before the first input arrives, a zero Twist is published so
  ``/cmd_vel`` is never stale/undefined.

The ``rclpy`` imports are gated with ``try/except ImportError`` so this
module imports cleanly on non-ROS hosts; the ROS wrapper is only
defined when ``rclpy`` is present. :func:`main` raises at runtime if
invoked without ROS.

Sign convention (pinned from ``serial_controller.py`` lines 123–124)::

    ("turn left",  -0.5, 0.3)   # negative steer → left turn in IRL
    ("turn right",  0.5, 0.3)   # positive steer → right turn in IRL

This adapter preserves that convention: ``angular_z > 0`` encodes a
right turn, matching IRL rather than REP-103. The Gazebo diff-drive
plugin is assumed to honour this; any per-robot correction is out of
scope for this adapter.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Tuning constants — operators can retune these in two ways:
#   1. Edit this file (affects the pure function used by tests and the
#      default ROS parameter values).
#   2. Pass ROS parameters ``max_linear_mps`` / ``max_angular_rps`` at
#      node launch — takes precedence over these module-level defaults at
#      runtime without touching the pure function.
# ---------------------------------------------------------------------------

#: Forward speed (m/s) that corresponds to full throttle (1.0).
#: This is a tuning knob — the value is deliberately not load-bearing for
#: the mapping shape; only the ratio steer/throttle → angular/linear
#: matters for path-following behaviour.
MAX_LINEAR_MPS: float = 1.5

#: Yaw rate (rad/s) that corresponds to full steer magnitude (1.0).
#: Set equal to ``MAX_LINEAR_MPS`` so the turning radius at full throttle
#: and full steer is 1.0 m (convenient default).
MAX_ANGULAR_RPS: float = 1.5


def steer_throttle_to_twist(
    steer: float,
    throttle: float,
    *,
    max_linear_mps: float = MAX_LINEAR_MPS,
    max_angular_rps: float = MAX_ANGULAR_RPS,
) -> tuple[float, float]:
    """Convert a ``(steer, throttle)`` action to ``(linear_x, angular_z)``.

    This is a pure, stateless function with no ROS dependency.

    The mapping is deliberately simple (linear) to give a transparent,
    debuggable baseline. Both inputs are clamped before scaling, which
    mirrors the clamping in ``serial_controller.SerialController.send``
    and keeps sim/IRL behaviour consistent.

    Parameters
    ----------
    steer:
        Steering command in ``[-1, 1]`` where ``-1`` is full left and
        ``+1`` is full right (IRL sign convention from
        ``serial_controller.py:67``). Values outside ``[-1, 1]`` are
        clamped before the linear map is applied.
    throttle:
        Throttle command in ``[0, 1]`` where ``0`` is stopped and ``1``
        is full speed. Negative values clamp to ``0`` (no reverse drive).
        Values above ``1`` clamp to ``1``.
    max_linear_mps:
        Forward speed (m/s) at ``throttle == 1.0``. Defaults to the
        module-level :data:`MAX_LINEAR_MPS` constant.
    max_angular_rps:
        Yaw rate (rad/s) at ``|steer| == 1.0``. Defaults to the
        module-level :data:`MAX_ANGULAR_RPS` constant.

    Returns
    -------
    tuple[float, float]
        ``(linear_x, angular_z)`` ready to populate a
        ``geometry_msgs/Twist``. ``linear_x`` is always ``>= 0``
        (no reverse); ``angular_z > 0`` encodes a right turn (positive
        steer, matching IRL convention).
    """
    # Clamp inputs to their valid domains — mirrors serial_controller.send().
    steer_clamped = max(-1.0, min(1.0, float(steer)))
    throttle_clamped = max(0.0, min(1.0, float(throttle)))

    linear_x = throttle_clamped * max_linear_mps
    angular_z = steer_clamped * max_angular_rps

    return linear_x, angular_z


# ---------------------------------------------------------------------------
# ROS 2 wrapper — gated so the module stays importable on bare hosts.
# ---------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist, Vector3

    ROS_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only off-ROS hosts
    ROS_AVAILABLE = False
    Node = object  # type: ignore[misc,assignment]


if ROS_AVAILABLE:

    class SteerThrottleToCmdVelNode(Node):  # type: ignore[misc,valid-type]
        """ROS 2 node that adapts ``(steer, throttle)`` Vector3 → ``Twist``.

        Subscribes to a ``geometry_msgs/Vector3`` topic where:
            ``x = steer  ∈ [-1, 1]``
            ``y = throttle ∈ [0, 1]``
            ``z = 0`` (unused)

        This layout is the exact format published by
        :class:`~couchmo_expert.waypoint_expert_node.WaypointExpertNode`
        and (later) the learned-policy inference node.

        The subscription callback caches the latest ``(steer, throttle)``
        pair. A timer fires at ``publish_rate_hz`` (default 10 Hz) and
        publishes a ``geometry_msgs/Twist`` on ``/cmd_vel``. Before the
        first input message arrives, a zero Twist is published so
        ``/cmd_vel`` is never stale or undefined.

        Parameters (declared on the node; set via YAML launch config):

        ``input_topic`` (str)
            Vector3 subscription topic (default ``/expert/steer_throttle``).
        ``output_topic`` (str)
            Twist publisher topic (default ``/cmd_vel``).
        ``publish_rate_hz`` (float)
            Timer rate in Hz (default 10.0).
        ``max_linear_mps`` (float)
            Forward speed at throttle == 1.0 (default :data:`MAX_LINEAR_MPS`).
            Tuning knob; does not affect the pure mapping function's
            module-level constant.
        ``max_angular_rps`` (float)
            Yaw rate at |steer| == 1.0 (default :data:`MAX_ANGULAR_RPS`).
        """

        def __init__(self) -> None:
            super().__init__("steer_throttle_to_cmd_vel")
            self.declare_parameter("input_topic", "/expert/steer_throttle")
            self.declare_parameter("output_topic", "/cmd_vel")
            self.declare_parameter("publish_rate_hz", 10.0)
            self.declare_parameter("max_linear_mps", MAX_LINEAR_MPS)
            self.declare_parameter("max_angular_rps", MAX_ANGULAR_RPS)

            input_topic = str(self.get_parameter("input_topic").value)
            output_topic = str(self.get_parameter("output_topic").value)
            rate_hz = float(self.get_parameter("publish_rate_hz").value)
            if rate_hz <= 0.0:
                raise ValueError(
                    f"publish_rate_hz must be > 0; got {rate_hz}"
                )
            self._max_linear_mps = float(
                self.get_parameter("max_linear_mps").value
            )
            self._max_angular_rps = float(
                self.get_parameter("max_angular_rps").value
            )

            # Cache of the latest steer/throttle received from the input topic.
            # Initialised to (0, 0) so the timer publishes safe zeros before
            # the first message arrives.
            self._steer: float = 0.0
            self._throttle: float = 0.0

            self._sub = self.create_subscription(
                Vector3, input_topic, self._on_steer_throttle, 10
            )
            self._pub = self.create_publisher(Twist, output_topic, 10)
            self._timer = self.create_timer(1.0 / rate_hz, self._on_tick)

            self.get_logger().info(
                f"steer_throttle_to_cmd_vel: subscribing {input_topic}; "
                f"publishing {output_topic} at {rate_hz:.1f} Hz; "
                f"max_linear_mps={self._max_linear_mps}, "
                f"max_angular_rps={self._max_angular_rps}"
            )

        def _on_steer_throttle(self, msg: "Vector3") -> None:
            """Cache the latest steer (msg.x) and throttle (msg.y)."""
            self._steer = float(msg.x)
            self._throttle = float(msg.y)

        def _on_tick(self) -> None:
            """Timer callback: publish Twist from cached steer/throttle."""
            linear_x, angular_z = steer_throttle_to_twist(
                self._steer,
                self._throttle,
                max_linear_mps=self._max_linear_mps,
                max_angular_rps=self._max_angular_rps,
            )
            cmd = Twist()
            cmd.linear.x = linear_x
            cmd.angular.z = angular_z
            self._pub.publish(cmd)


def main(args: list[str] | None = None) -> None:
    """Entry point for the ``steer_throttle_to_cmd_vel`` console script."""
    if not ROS_AVAILABLE:
        raise RuntimeError(
            "rclpy not available; this node must run inside the sim container"
        )
    rclpy.init(args=args)
    node = SteerThrottleToCmdVelNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
