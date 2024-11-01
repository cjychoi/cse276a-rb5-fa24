from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='hw2_yolo_camera_node',
            name='hw2_yolo_camera_node',
            output='screen'
        ),
        Node(
            package='my_package',
            executable='hw2_pid_controller_node',
            name='hw2_pid_controller_node',
            output='screen'
        ),
        Node(
            package='my_package',
            executable='megapi_controller_node',
            name='megapi_controller_node',
            output='screen'
        ),
    ])
