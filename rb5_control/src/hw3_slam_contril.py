# slam_control_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import matplotlib.pyplot as plt
import time

# Placeholder for EKF SLAM class
class EKFSLAM:
    def __init__(self, object_list):
        # Initialize SLAM state (robot pose and landmarks)
        self.state = np.zeros((3 + 2 * len(object_list), 1))

    def update(self, measurement, obj_index):
        # Placeholder for EKF update
        pass

    def get_state(self):
        return self.state

class SlamControlNode(Node):
    def __init__(self):
        super().__init__('slam_control_node')

        # Publisher for robot motion (Twist)
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)

        # Subscriber for object detection info
        self.subscription = self.create_subscription(
            Float32MultiArray, '/detected_object_info', self.object_callback, 10
        )

        # Initialize EKF SLAM
        self.objects_to_detect = ['teddy bear', 'backpack', 'umbrella', 'bottle']
        self.ekf_slam = EKFSLAM(self.objects_to_detect)

        # Initialize plot for real-time visualization
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-5, 5)  # Adjust limits based on environment size
        self.ax.set_ylim(-5, 5)
        self.robot_line, = self.ax.plot([], [], 'bo', label="Robot")
        self.robot_positions = []

        # Start the movement after subscribing
        self.spin_and_track()

    def object_callback(self, msg):
        distance, angle = msg.data
        obj_index = 0  # Placeholder for the detected object's index
        self.ekf_slam.update(np.array([distance, angle]), obj_index)

    def spin_and_track(self):
        # Move the robot in a 2m x 2m square
        for i in range(4):  # For a square
            self.move_forward(2.0)  # Move 2.0 meters
            self.turn_90_degrees()  # Turn 90 degrees

        # After completing the square, stop the robot and plot its path
        self.plot_robot_positions()

        # Save the plot
        self.save_plot()

    def move_forward(self, distance):
        # Move the robot forward by a specified distance (in meters)
        move_twist = Twist()
        move_twist.linear.x = 2.0  # Set a faster forward speed
        self.publisher_.publish(move_twist)
        time.sleep(distance / 1.0 * 4)  # Move for the time required based on speed
        move_twist.linear.x = 0.0  # Stop the robot
        self.publisher_.publish(move_twist)

    def turn_90_degrees(self):
        # Rotate the robot 90 degrees
        turn_twist = Twist()
        turn_twist.angular.z = 9.0  # Set a faster rotation speed
        self.publisher_.publish(turn_twist)
        time.sleep(1.57)  # 90 degrees = 1.57 radians, so it takes 1.57 seconds at 1.0 rad/s
        turn_twist.angular.z = 0.0  # Stop rotating
        self.publisher_.publish(turn_twist)

    def plot_robot_positions(self):
        # Update the robot's position and plot it
        state = self.ekf_slam.get_state()
        robot_x, robot_y, _ = state[0, 0], state[1, 0], state[2, 0]
        self.robot_positions.append([robot_x, robot_y])  # Add the robot's new position to the list

        # Plot all robot positions together in blue
        robot_positions_array = np.array(self.robot_positions)
        self.robot_line.set_data(robot_positions_array[:, 0], robot_positions_array[:, 1])

    def save_plot(self):
        self.fig.savefig('slam_plot.png')  # Save the plot as an image file


def main(args=None):
    rclpy.init(args=args)
    slam_node = SlamControlNode()

    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass

    slam_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
