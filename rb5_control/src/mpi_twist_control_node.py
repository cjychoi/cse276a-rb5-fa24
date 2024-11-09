#!/usr/bin/env python3
# MegaPi Controller ROS2 Wrapper
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from mpi_control import MegaPiController
import numpy as np


class MegaPiControllerNode(Node):
    def __init__(self, verbose=True, debug=False):
        super().__init__("megapi_controller_node")
        self.mpi_ctrl = MegaPiController(port="/dev/ttyUSB0", verbose=verbose)
        self.r = 0.025  # radius of the wheel
        self.lx = 0.055  # half of the distance between front and back wheels
        self.ly = 0.07  # half of the distance between left and right wheels
        self.calibration = 1.0

        # Robot pose state variables
        self.x = 0.0  # Initial x position
        self.y = 0.0  # Initial y position
        self.theta = 0.0  # Initial orientation (theta in radians)
        self.last_update_time = self.get_clock().now()  # Track the last update time

        # Subscribe to Twist messages for controlling the robot
        self.subscription = self.create_subscription(
            Twist, "/twist", self.twist_callback, 10
        )
        self.subscription

        # Print robot pose every 0.5 seconds
        self.create_timer(0.5, self.print_pose)

    def twist_callback(self, twist_cmd):
        current_time = self.get_clock().now()
        delta_time = (current_time - self.last_update_time).nanoseconds * 1e-9
        self.last_update_time = current_time

        # Calculate the change in position and orientation
        delta_x = twist_cmd.linear.x * delta_time * np.cos(self.theta)
        delta_y = twist_cmd.linear.x * delta_time * np.sin(self.theta)
        delta_theta = twist_cmd.angular.z * delta_time

        self.theta += delta_theta
        self.theta = self.theta % (2 * np.pi)

        # Update robot pose
        self.x += delta_x
        self.y += delta_y

        desired_twist = self.calibration * np.array(
            [[twist_cmd.linear.x], [twist_cmd.linear.y], [twist_cmd.angular.z]]
        )
        print(desired_twist)

        # Calculate the Jacobian matrix
        jacobian_matrix = (
            np.array(
                [
                    [1, -1, -(self.lx + self.ly)],
                    [1, 1, (self.lx + self.ly)],
                    [1, 1, -(self.lx + self.ly)],
                    [1, -1, (self.lx + self.ly)],
                ]
            )
            / self.r
        )

        # Calculate the desired wheel velocity
        result = np.dot(jacobian_matrix, desired_twist)

        # Send command to each wheel
        self.mpi_ctrl.setFourMotors(
            int(result[0][0]), int(result[1][0]), int(result[2][0]), int(result[3][0])
        )

    def print_pose(self):
        # Print the robot's pose (x, y, theta) every 0.5 seconds
        print(f"Robot pose: x = {self.x:.2f}, y = {self.y:.2f}, theta = {self.theta:.2f} radians")


def main(args=None):
    rclpy.init(args=args)
    mpi_ctrl_node = MegaPiControllerNode()

    try:
        rclpy.spin(mpi_ctrl_node)
    except KeyboardInterrupt:
        pass

    mpi_ctrl_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
