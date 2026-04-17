"""ROS 2 launch file for the dataset-recording stack.

Starts two nodes side-by-side:

1. ``waypoint_expert_node`` — runs the pure-pursuit expert and publishes
   ``(steer, throttle)`` on ``/expert/steer_throttle``.
2. ``dataset_recorder_node`` — synchronises the two camera images with the
   expert actions and writes ``.npz`` shards to the bind-mounted data volume.

This launch file does **not** start Gazebo — use ``gazebo.launch.py`` for
that.  The expectation is that Gazebo (and therefore the camera and odometry
topics) is already running when this file is invoked.

Typical usage (inside the sim container, after ``colcon build``)::

    ros2 launch couchmo_expert record_dataset.launch.py \\
        episode_id:=ep_20260416_181234 \\
        waypoints_csv:=/workspace/data/campus.csv

All other arguments have sensible defaults matching the topic names set in
Tasks 2 (cameras) and 5 (waypoint expert).
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Build the launch description for the dataset-recording stack."""
    pkg_share = Path(get_package_share_directory("couchmo_expert"))
    default_config = pkg_share / "config" / "waypoint_follower.yaml"

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------

    episode_id_arg = DeclareLaunchArgument(
        "episode_id",
        description=(
            "Unique identifier for this recording run, e.g. "
            "'ep_20260416_181234'. Required — two runs with the same "
            "episode_id would collide on disk."
        ),
    )

    waypoints_csv_arg = DeclareLaunchArgument(
        "waypoints_csv",
        description=(
            "Absolute path to the waypoint CSV (x_m,y_m,s_m) produced by "
            "kml_to_waypoints.py. Required by the waypoint expert."
        ),
    )

    output_root_arg = DeclareLaunchArgument(
        "output_root",
        default_value="/workspace/data",
        description=(
            "Root directory where episode shards are written. "
            "Shards land at <output_root>/<episode_id>/shard_<n>.npz. "
            "Host-visible via the bind mount declared in docker-compose.yml."
        ),
    )

    world_arg = DeclareLaunchArgument(
        "world",
        default_value="",
        description="Gazebo world name, recorded in manifest.json (informational).",
    )

    notes_arg = DeclareLaunchArgument(
        "notes",
        default_value="",
        description="Free-text notes recorded in manifest.json (informational).",
    )

    samples_per_shard_arg = DeclareLaunchArgument(
        "samples_per_shard",
        default_value="200",
        description="Number of samples per .npz shard before a new shard is started.",
    )

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    # Mirror the parameter-loading pattern from waypoint_expert.launch.py:
    # load the shared YAML tuning config first, then override with launch args.
    waypoint_expert_node = Node(
        package="couchmo_expert",
        executable="waypoint_expert_node",
        name="waypoint_expert",
        output="screen",
        parameters=[
            str(default_config),
            {"waypoints_csv": LaunchConfiguration("waypoints_csv")},
        ],
    )

    dataset_recorder_node = Node(
        package="couchmo_expert",
        executable="dataset_recorder_node",
        name="dataset_recorder",
        output="screen",
        parameters=[
            {
                "episode_id": LaunchConfiguration("episode_id"),
                "output_root": LaunchConfiguration("output_root"),
                "world": LaunchConfiguration("world"),
                "notes": LaunchConfiguration("notes"),
                "samples_per_shard": LaunchConfiguration("samples_per_shard"),
            }
        ],
    )

    return LaunchDescription(
        [
            # Args first — consumed by nodes below
            episode_id_arg,
            waypoints_csv_arg,
            output_root_arg,
            world_arg,
            notes_arg,
            samples_per_shard_arg,
            # Nodes
            waypoint_expert_node,
            dataset_recorder_node,
        ]
    )
