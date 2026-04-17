"""ROS 2 launch file for the pure-pursuit waypoint expert.

Typical usage (inside the sim container, after ``colcon build``)::

    ros2 launch couchmo_expert waypoint_expert.launch.py \\
        waypoints_csv:=/abs/path/to/campus.csv

The launch file loads ``config/waypoint_follower.yaml`` from this
package's share directory for all tuning parameters, then overrides
``waypoints_csv`` with the mandatory launch argument of the same name.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Build the launch description for ``waypoint_expert_node``."""
    pkg_share = Path(get_package_share_directory("couchmo_expert"))
    default_config = pkg_share / "config" / "waypoint_follower.yaml"

    waypoints_csv_arg = DeclareLaunchArgument(
        "waypoints_csv",
        description=(
            "Absolute path to the waypoint CSV (x_m,y_m,s_m) produced by "
            "kml_to_waypoints.py. Required — no default."
        ),
    )

    expert_node = Node(
        package="couchmo_expert",
        executable="waypoint_expert_node",
        name="waypoint_expert",
        output="screen",
        parameters=[
            str(default_config),
            {"waypoints_csv": LaunchConfiguration("waypoints_csv")},
        ],
    )

    return LaunchDescription([waypoints_csv_arg, expert_node])
