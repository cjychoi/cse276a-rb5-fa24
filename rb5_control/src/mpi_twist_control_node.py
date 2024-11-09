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

        # Subscribe to Twist messages for controlling the robot
        self.subscription = self.create_subscription(
            Twist, "/twist", self.twist_callback, 10
        )
        self.subscription

    def twist_callback(self, twist_cmd):
        desired_twist = self.calibration * np.array(
            [[twist_cmd.linear.x], [twist_cmd.linear.y], [twist_cmd.angular.z]]
        )
        print(desired_twist)

        # Calculate the jacobian matrix
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
