# Working version of waypoint_navigator_script2.py

import time
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from mpi_control import MegaPiController
import matplotlib.pyplot as plt

class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')
        self.get_logger().info("Starting waypoint navigator...")

        # Initialize the MegaPiController
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)  # Allow some time for the connection to establish

        # Control parameters
        self.k_v = 30  # Speed for straight movement
        self.k_w = 55  # Speed for rotational movement
        self.dist_per_sec = 10 / 1  # 10 cm per 1 second at speed 30
        self.rad_per_sec = math.pi / 2  # Pi radians per 2 seconds at speed 55

        # Robot state variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0  # Robot's orientation in radians

        # Plotting variables
        self.x_history = [self.current_x]
        self.y_history = [self.current_y]

        # Subscribe to rotation and movement commands from the YoloCameraNode
        self.rotation_sub = self.create_subscription(
            Float32,
            '/rotation_angle',
            self.handle_rotation,
            10
        )
        self.move_sub = self.create_subscription(
            Float32,
            '/move_distance',
            self.handle_movement,
            10
        )

        # Initialize the plot for the robot's movement
        self.fig, self.ax = plt.subplots()
        self.plot_robot_movement()

    def plot_robot_movement(self):
        # Update the plot with the current trajectory
        self.ax.clear()
        self.ax.plot(self.x_history, self.y_history, marker='o')
        self.ax.set_xlim(-10, 10)  # Set limits according to the movement range
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('Robot Trajectory')
        plt.pause(0.1)  # Update plot every 0.1 seconds

    def handle_rotation(self, msg):
        self.get_logger().info(f"11111111")
        print("handle_rotation called")
        angle_diff = msg.data
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.get_logger().info(f"2222222")
        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        self.get_logger().info(f"3333333")
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

        # Update the robot's orientation after the rotation
        self.current_theta += angle_diff
        self.current_theta %= (2 * math.pi)  # Keep theta in [0, 2*pi] range

        self.get_logger().info(f"Rotated by {angle_diff} radians.")
        self.plot_robot_movement()

    def handle_movement(self, msg):
        print("handle_movement called")
        distance = msg.data
        movement_time = abs(distance) / (self.dist_per_sec / 100)  # Convert cm/s to m/s
        self.mpi_ctrl.carStraight(self.k_v)
        time.sleep(movement_time)
        self.mpi_ctrl.carStop()

        # Update the robot's position after moving forward
        delta_x = distance * math.cos(self.current_theta) / 100.0  # Convert cm to meters
        delta_y = distance * math.sin(self.current_theta) / 100.0  # Convert cm to meters
        self.current_x += delta_x
        self.current_y += delta_y

        self.x_history.append(self.current_x)
        self.y_history.append(self.current_y)

        self.get_logger().info(f"Moved forward by {distance} cm.")
        self.plot_robot_movement()

    def main_loop(self):
        # Keep the node running and plotting
        while rclpy.ok():
            rclpy.spin_once(self)
            self.plot_robot_movement()

def main(args=None):
    rclpy.init(args=args)
    navigator = WaypointNavigator()

    try:
        navigator.main_loop()
    except KeyboardInterrupt:
        navigator.get_logger().info("Shutting down...")

    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()