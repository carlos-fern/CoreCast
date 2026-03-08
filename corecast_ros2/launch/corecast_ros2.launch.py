import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    params_file = os.path.join(
        get_package_share_directory("corecast_ros2"),
        "config",
        "params.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="corecast_ros2",
                executable="corecast_ros2_node",
                name="corecast_ros2_node",
                output="screen",
                parameters=[params_file],
            )
        ]
    )
