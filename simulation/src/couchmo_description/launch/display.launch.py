"""
display.launch.py
RViz2 visualization for URDF inspection (no Gazebo).
"""

import os
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("couchmo_description")
    urdf_file = os.path.join(pkg, "urdf", "couchmo.urdf.xacro")
    rviz_config = os.path.join(pkg, "rviz", "couchmo.rviz")

    result = subprocess.run(
        ["xacro", urdf_file], capture_output=True, text=True, check=True
    )
    robot_description_str = result.stdout

    # robot_state_publisher receives the description as a parameter
    # AND republishes it on the /robot_description topic (latched) for RViz
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description_str,
            "publish_frequency": 30.0,
        }],
    )

    # Static zero joint states for the two continuous wheel joints
    publish_joint_states = ExecuteProcess(
        cmd=[
            "ros2", "topic", "pub", "--rate", "10",
            "/joint_states",
            "sensor_msgs/msg/JointState",
            (
                '{"header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""}, '
                '"name": ["left_wheel_joint", "right_wheel_joint"], '
                '"position": [0.0, 0.0], '
                '"velocity": [0.0, 0.0], '
                '"effort": [0.0, 0.0]}'
            ),
        ],
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
    )

    return LaunchDescription([
        robot_state_publisher,
        publish_joint_states,
        rviz,
    ])
