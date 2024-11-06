import time
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from mpi_control import MegaPiController
import matplotlib.pyplot as plt
import cv2
import numpy as np


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

        # Robot state variables (position and orientation)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0  # Robot's orientation in radians
        self.is_moving = False  # Flag to indicate if the robot is moving
        self.total_distance = 0.0  # Variable to track total distance traveled

        # Plotting variables
        self.x_history = [self.current_x]
        self.y_history = [self.current_y]

        # EKF state (tracking position, orientation, and landmarks)
        self.ekf_state = np.array([self.current_x, self.current_y, self.current_theta])
        self.P = np.eye(3) * 1e-3  # Initial covariance matrix
        self.Q = np.diag([0.1, 0.1, np.deg2rad(5)]) ** 2  # Process noise covariance
        self.R = np.diag([0.2, np.deg2rad(2)]) ** 2  # Measurement noise covariance
        self.landmarks = {}  # Store the mapped landmarks
        self.landmark_covariances = {}  # Store the uncertainties of landmarks

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
        self.save_plot_sub = self.create_subscription(
            Float32,
            '/save_plot',
            self.save_plot_callback,
            10
        )

        # Publisher for is_moving flag
        self.is_moving_pub = self.create_publisher(Bool, '/is_moving', 10)

        # Initialize the plot for the robot's movement
        self.fig, self.ax = plt.subplots()
        self.plot_robot_movement()

    def plot_robot_movement(self):
        # Update the plot with the current trajectory
        self.ax.clear()
        self.ax.plot(self.x_history, self.y_history, marker='o')
        
        # Plot landmarks (if any)
        for name, (lx, ly) in self.landmarks.items():
            self.ax.plot(lx, ly, 'rx', label=f'Landmark: {name}')
        
        self.ax.set_xlim(-10, 10)  # Set limits according to the movement range
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title(f'Robot Trajectory - Total Distance: {self.total_distance:.2f} cm')
        plt.pause(0.1)  # Update plot every 0.1 seconds

    def save_plot_callback(self, msg):
        # Save the plot as a JPEG file
        plot_filename = 'robot_trajectory.jpeg'
        self.fig.savefig(plot_filename, format='jpeg')
        img = cv2.imread(f'{plot_filename}')
        cv2.imshow('Robot Trajectory', img)
        self.get_logger().info(f"Plot saved as {plot_filename}")

    def handle_rotation(self, msg):
        angle_diff = msg.data
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.is_moving = True  # Set the moving flag to True
        self.is_moving_pub.publish(Bool(data=True))  # Notify that the robot is moving

        # Send the rotate command to the robot
        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

        # Update the robot's orientation after the rotation
        self.current_theta += angle_diff
        self.current_theta %= (2 * math.pi)  # Keep theta in [0, 2*pi] range

        # EKF prediction step (rotation only)
        self.ekf_predict(0, angle_diff)

        self.get_logger().info(f"Rotated by {angle_diff} radians.")
        
        self.is_moving = False  # Set the moving flag to False
        self.is_moving_pub.publish(Bool(data=False))  # Notify that the robot is no longer moving
        self.plot_robot_movement()

    def handle_movement(self, msg):
        distance = msg.data
        movement_time = abs(distance) / self.dist_per_sec  # Convert cm/s to m/s
        self.is_moving = True  # Set the moving flag to True
        self.is_moving_pub.publish(Bool(data=True))  # Notify that the robot is moving

        # Send the move forward command to the robot
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

        # EKF prediction step (forward movement)
        self.ekf_predict(distance, 0)

        # Update total distance traveled
        self.total_distance += distance

        self.get_logger().info(f"Moved forward by {distance} cm.")
        
        self.is_moving = False  # Set the moving flag to False
        self.is_moving_pub.publish(Bool(data=False))  # Notify that the robot is no longer moving
        self.plot_robot_movement()

    def ekf_predict(self, distance, angle_diff):
        """
        EKF prediction step for robot motion (both translation and rotation).
        """
        # Update EKF state based on motion model
        if distance != 0:
            dx = distance * math.cos(self.ekf_state[2])
            dy = distance * math.sin(self.ekf_state[2])
            self.ekf_state[0] += dx / 100  # Convert from cm to meters
            self.ekf_state[1] += dy / 100  # Convert from cm to meters
        
        if angle_diff != 0:
            self.ekf_state[2] += angle_diff
            self.ekf_state[2] %= (2 * math.pi)  # Normalize orientation

        # Compute the Jacobian of the motion model
        F = np.eye(3)
        if distance != 0:
            F[0, 2] = -distance * math.sin(self.ekf_state[2])
            F[1, 2] = distance * math.cos(self.ekf_state[2])

        # Update the covariance matrix with process noise
        self.P = F @ self.P @ F.T + self.Q

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

    navigator.save_plot_callback(Float32())  # Save the plot on shutdown
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
