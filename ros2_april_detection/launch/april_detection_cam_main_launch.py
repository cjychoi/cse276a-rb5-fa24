from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rb5_vision_dir = get_package_share_directory("rb5_ros2_vision")
    rb_camera_main_ocv_launch = os.path.join(
        rb5_vision_dir, "launch", "rb_camera_main_ocv_launch.py"
    )

    return LaunchDescription(
        [
            # Including another launch file
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rb_camera_main_ocv_launch)
            ),
            # April detection node
            Node(
                package="ros2_april_detection",
                executable="april_detection_node",
                name="april_detection_node",
                output="screen",
            ),
            # if you want to use tf2, uncomment and modify the followng Nodes to match your environment
            # Static transform publisher for marker1
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="marker1_trans",
                arguments=[
                    "-0.5",
                    "2.0",
                    "0.0",
                    "-0.5",
                    "-0.5",
                    "0.5",
                    "0.5",
                    "map",
                    "marker_1",
                ],
            ),
            # Static transform publisher for marker0
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="marker0_trans",
                arguments=[
                    "1.34",
                    "0.0",
                    "0.0",
                    "0.5",
                    "-0.5",
                    "0.5",
                    "-0.5",
                    "map",
                    "marker_0",
                ],
            ),
        ]
    )
