#!/usr/bin/env python3
""" MegaPi Controller ROS2 Wrapper"""
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from mpi_control import MegaPiController
import numpy as np

class MegaPiControllerNode(Node):
    def __init__(self, verbose=False, debug=False):
        super().__init__('megapi_controller_node')
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=verbose)

    # This is a ROS2 Foxy template for MegaPi controller




if __name__ == "__main__":
    rclpy.init()
    mpi_ctrl_node = MegaPiControllerNode()

    rclpy.spin(mpi_ctrl_node) # Spin for until 
    
    # Destroy node and shutdown when done. (Optional, as node would be cleaned up by garbage collection)
    mpi_ctrl_node.destroy_node()
    rclpy.shutdown()