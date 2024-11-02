import time 
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from mpi_control import MegaPiController
import matplotlib.pyplot as plt
import cv2

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
        self.is_moving = False  # Flag to indicate if the robot is moving
        self.total_distance = 0.0  # Variable to track total distance traveled

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
        self.save_plot = self.create_subscription(
            Float32,
            '/save_plot',
            self.save_plot,
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
        self.ax.set_xlim(-10, 10)  # Set limits according to the movement range
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title(f'Robot Trajectory - Total Distance: {self.total_distance:.2f} cm')
        plt.pause(0.1)  # Update plot every 0.1 seconds

    def save_plot(self, msg):
        # Save the plot as a JPEG file
        plot_filename = 'robot_trajectory.jpeg'
        self.fig.savefig(plot_filename, format='jpeg')
        img = cv2.imread(f'{plot_filename}.jpg')

        # Display the image
        cv2.imshow('Image', img)
        self.get_logger().info(f"Plot saved as {plot_filename}")

    def handle_rotation(self, msg):
        angle_diff = msg.data
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.is_moving = True  # Set the moving flag to True
        self.is_moving_pub.publish(Bool(data=True))  # Notify that the robot is moving

        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

        # Update the robot's orientation after the rotation
        self.current_theta += angle_diff
        self.current_theta %= (2 * math.pi)  # Keep theta in [0, 2*pi] range

        self.get_logger().info(f"Rotated by {angle_diff} radians.")
        
        self.is_moving = False  # Set the moving flag to False
        self.is_moving_pub.publish(Bool(data=False))  # Notify that the robot is no longer moving
        self.plot_robot_movement()

    def handle_movement(self, msg):
        distance = msg.data
        movement_time = abs(distance) / self.dist_per_sec  # Convert cm/s to m/s
        self.is_moving = True  # Set the moving flag to True
        self.is_moving_pub.publish(Bool(data=True))  # Notify that the robot is moving

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

        # Update total distance traveled
        self.total_distance += distance

        self.get_logger().info(f"Moved forward by {distance} cm.")
        
        self.is_moving = False  # Set the moving flag to False
        self.is_moving_pub.publish(Bool(data=False))  # Notify that the robot is no longer moving
        self.plot_robot_movement()

    def main_loop(self):
        # Keep the node running and plotting
        while rclpy.ok():
            rclpy.spin_once(self)
            self.plot_robot_movement()
            # Call save_plot at some point if needed, for example after all movements
            # self.save_plot()  # Uncomment this line to save the plot after each iteration if desired

def main(args=None):
    rclpy.init(args=args)
    navigator = WaypointNavigator()

    try:
        navigator.main_loop()
        # Uncomment to save the plot when shutting down
        # navigator.save_plot()  # Save the plot when shutting down
    except KeyboardInterrupt:
        navigator.get_logger().info("Shutting down...")

    navigator.save_plot()  # Save the plot on shutdown
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
