import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    params_file = os.path.join(
        get_package_share_directory("corecast_ros2"),
        "config",
        "params.yaml",
    )
    static_pointcloud = LaunchConfiguration("static_pointcloud")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "static_pointcloud",
                default_value="true",
                description="If true, keep pointcloud static while TF frame moves.",
            ),
            Node(
                package="corecast_ros2",
                executable="corecast_ros2_node",
                name="corecast_ros2_node",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="corecast_ros2",
                executable="corecast_ros2_pointcloud_sim_node",
                name="corecast_ros2_pointcloud_sim_node",
                output="screen",
                parameters=[{"static_pointcloud": static_pointcloud}],
            ),
            Node(
                package="foxglove_bridge",
                executable="foxglove_bridge",
                name="foxglove_bridge",
                output="screen",
            ),
        ]
    )
