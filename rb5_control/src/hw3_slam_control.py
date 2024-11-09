# slam_control_node.py - Modified with debug prints and fixed plotting
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
        self.object_list = object_list
        self.colors = plt.cm.get_cmap('tab10', len(object_list))
    
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
        self.objects_to_detect = ['teddy bear', 'backpack', 'umbrella', 'bottle', 'stop sign', 'car']
        self.ekf_slam = EKFSLAM(self.objects_to_detect)

        # Initialize plot for real-time visualization
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-5, 5)  # Adjust limits based on environment size
        self.ax.set_ylim(-5, 5)
        self.robot_positions = []
        self.obj_positions = {obj: [] for obj in self.objects_to_detect}
        self.legend_added = False

        # Start the movement after subscribing
        self.spin_and_track()

    def object_callback(self, msg):
        distance, angle = msg.data
        obj_index = 0  # Placeholder for the detected object's index
        self.ekf_slam.update(np.array([distance, angle]), obj_index)

    def spin_and_track(self):
        # Move the robot in a 2m x 2m square, stopping every 0.5 meters or every 45 degrees
        time.sleep(2)
        for _ in range(4):  # For a square
            for _ in range(4):  # Stop every 0.5 meters
                self.move_forward(0.5)  
                self.update_and_plot()
            for _ in range(2):  # Stop every 45 degrees in a 90-degree turn
                self.turn_45_degrees()  
                self.update_and_plot()

        # After completing the square, stop the robot and plot its path
        self.plot_robot_positions()

        # Save the plot
        self.save_plot()

    def move_forward(self, distance):
        # Move the robot forward by a specified distance (in meters)
        print("Moving forward by 0.5 meters")
        move_twist = Twist()
        move_twist.linear.x = 2.0  # Set slower speed for controlled movement
        self.publisher_.publish(move_twist)
        time.sleep(distance / 0.5)  # Move for the time required based on speed
        move_twist.linear.x = 0.0  # Stop the robot
        self.publisher_.publish(move_twist)

    def turn_45_degrees(self):
        # Rotate the robot 45 degrees
        print("Turning 45 degrees")
        turn_twist = Twist()
        turn_twist.angular.z = 9.0  # Set rotation speed
        self.publisher_.publish(turn_twist)
        time.sleep(np.pi / 4)  # 45 degrees = pi/4 radians
        turn_twist.angular.z = 0.0  # Stop rotating
        self.publisher_.publish(turn_twist)

    def update_and_plot(self):
        # Update the robot's position and plot it along with detected objects
        state = self.ekf_slam.get_state()
        robot_x, robot_y, theta = state[0, 0], state[1, 0], state[2, 0]
        self.robot_positions.append([robot_x, robot_y])
        
        # Clear and replot the robot path to prevent duplicated points in legend
        self.ax.clear()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.plot(*zip(*self.robot_positions), 'bo-', label="Robot Path")
        
        # Plot objects with unique colors
        for idx, obj_name in enumerate(self.objects_to_detect):
            if self.obj_positions[obj_name]:
                obj_pos_array = np.array(self.obj_positions[obj_name])
                self.ax.plot(obj_pos_array[:, 0], obj_pos_array[:, 1], 'o', color=self.ekf_slam.colors(idx), label=obj_name)

        # Draw legend only once
        if not self.legend_added:
            self.ax.legend()
            self.legend_added = True

        plt.pause(0.01)

    def plot_robot_positions(self):
        plt.title("Robot Path and Detected Object Positions")
        plt.xlabel("X position (meters)")
        plt.ylabel("Y position (meters)")

    def save_plot(self):
        self.fig.savefig('slam_plot.png')  # Save the plot as an image file
        print("Plot saved as slam_plot.png")

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