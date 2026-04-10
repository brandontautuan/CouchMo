"""
gazebo.launch.py
Gazebo Classic 11 + ROS 2 Humble — spawns Couchmo and bridges all sensor topics.
"""

import os
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg      = get_package_share_directory("couchmo_description")
    gz_pkg   = get_package_share_directory("gazebo_ros")
    world_name = LaunchConfiguration("world", default="folsom_lake_college")
    # Map world name → file path
    import sys
    _world_arg = [a for a in sys.argv if a.startswith("world:=")]
    _wname = _world_arg[0].split(":=")[1] if _world_arg else "folsom_lake_college"
    world    = os.path.join(pkg, "worlds", f"{_wname}.world")
    urdf     = os.path.join(pkg, "urdf", "couchmo.urdf.xacro")
    rviz_cfg = os.path.join(pkg, "rviz", "couchmo_nav.rviz")

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    # Process xacro eagerly — same pattern that works in display.launch.py
    robot_description_str = subprocess.run(
        ["xacro", urdf], capture_output=True, text=True, check=True
    ).stdout

    # ── Gazebo Classic ──────────────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_pkg, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "world":   world,
            "verbose": "false",
            "pause":   "false",
        }.items(),
    )

    # ── Robot state publisher ────────────────────────────────
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description_str,
            "use_sim_time":      use_sim_time,
        }],
    )

    # ── Spawn into Gazebo ────────────────────────────────────
    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "couchmo",
            "-x",  "0.0",   # campus main entry road
            "-y", "-50.0",  # south end, heading north
            "-z",  "0.15",
        ],
        output="screen",
    )

    # ── RViz (navigation view) ───────────────────────────────
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_cfg],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("world", default_value="folsom_lake_college",
                              description="World name (no .world extension)"),
        gazebo,
        robot_state_publisher,
        spawn,
        rviz,
    ])
