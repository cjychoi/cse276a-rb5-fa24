# hw3_slam_control.py - Updated to integrate EKF SLAM for all detected objects
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import matplotlib.pyplot as plt
import time

class EKFSLAM:
    def __init__(self, object_list):
        # Initialize SLAM state (robot pose and landmarks)
        self.state = np.zeros((3 + 2 * len(object_list), 1))  # [x, y, theta, x1, y1, x2, y2, ...]
        self.P = np.eye(3 + 2 * len(object_list)) * 1000  # Large initial uncertainty for landmarks
        self.R = np.diag([0.1, 0.1])  # Measurement noise for range and bearing
        self.object_list = object_list
        self.colors = plt.cm.get_cmap('tab10', len(object_list))

    def update(self, measurement, obj_index):
        # Extract the robot's position and orientation from the state
        x, y, theta = self.state[0, 0], self.state[1, 0], self.state[2, 0]
        
        # Extract the distance and bearing measurement to the landmark
        distance, bearing = measurement
        
        # Compute the landmark's index in the state vector
        landmark_idx = 3 + 2 * obj_index
        
        # Check if the landmark is already initialized
        if self.P[landmark_idx, landmark_idx] > 999:  # High initial uncertainty for uninitialized landmarks
            # Initialize the landmark position based on the robot's position and the measurement
            self.state[landmark_idx, 0] = x + distance * np.cos(theta + bearing)
            self.state[landmark_idx + 1, 0] = y + distance * np.sin(theta + bearing)
            self.P[landmark_idx:landmark_idx + 2, landmark_idx:landmark_idx + 2] = np.eye(2) * 100  # Initial landmark uncertainty

        # Predicted measurement for the landmark
        delta_x = self.state[landmark_idx, 0] - x
        delta_y = self.state[landmark_idx + 1, 0] - y
        q = delta_x**2 + delta_y**2
        predicted_distance = np.sqrt(q)
        predicted_bearing = np.arctan2(delta_y, delta_x) - theta
        predicted_measurement = np.array([[predicted_distance], [predicted_bearing]])
        
        # Measurement innovation (difference between actual and predicted measurements)
        innovation = np.array([[distance - predicted_distance], [bearing - predicted_bearing]])
        innovation[1, 0] = (innovation[1, 0] + np.pi) % (2 * np.pi) - np.pi  # Normalize bearing

        # Compute the Jacobian of the measurement function with respect to the state
        H = np.zeros((2, len(self.state)))
        H[0, 0] = -delta_x / predicted_distance
        H[0, 1] = -delta_y / predicted_distance
        H[1, 0] = delta_y / q
        H[1, 1] = -delta_x / q
        H[0, landmark_idx] = delta_x / predicted_distance
        H[0, landmark_idx + 1] = delta_y / predicted_distance
        H[1, landmark_idx] = -delta_y / q
        H[1, landmark_idx + 1] = delta_x / q

        # Compute the innovation covariance
        S = H @ self.P @ H.T + self.R

        # Compute the Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update the state and covariance matrix
        self.state += K @ innovation
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P

    def get_state(self):
        return self.state

class SlamControlNode(Node):
    def __init__(self):
        super().__init__('slam_control_node')
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)
        self.subscription = self.create_subscription(
            Float32MultiArray, '/detected_object_info', self.object_callback, 10
        )
        self.objects_to_detect = ['stop sign', 'car', 'teddy bear', 'bottle', 'backpack', 'umbrella']
        self.ekf_slam = EKFSLAM(self.objects_to_detect)
        self.fig, self.ax = plt.subplots()
        self.set_plot_limits()
        self.robot_positions = []  # Store estimated robot positions from EKF
        self.detections = []  # List to store EKF-predicted object positions
        self.spin_and_track()

    def set_plot_limits(self):
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)

    def object_callback(self, msg):
        # Extract the detection info
        distance, angle, obj_index = msg.data
        measurement = [distance, angle]
        
        # Update EKF SLAM with the measurement
        self.ekf_slam.update(measurement, int(obj_index))
        
        # Update the robot position with EKF estimate
        robot_x, robot_y = self.ekf_slam.state[0, 0], self.ekf_slam.state[1, 0]
        self.robot_positions.append([robot_x, robot_y])

        # Get the EKF-predicted object position
        landmark_idx = 3 + 2 * obj_index
        obj_x, obj_y = self.ekf_slam.state[landmark_idx, 0], self.ekf_slam.state[landmark_idx + 1, 0]
        object_name = self.objects_to_detect[obj_index]
        self.detections.append((obj_x, obj_y, object_name))
        
        # Update the plot
        self.update_and_plot()

    def update_and_plot(self):
        self.ax.clear()
        self.set_plot_limits()
        self.ax.plot(*zip(*self.robot_positions), 'bo-', label="Robot Path")

        # Create a dictionary to store unique labels for the legend
        legend_labels = {}

        # Plot each detected object based on EKF prediction
        for x, y, name in self.detections:
            color = self.ekf_slam.colors(self.objects_to_detect.index(name))
            if name not in legend_labels:
                legend_labels[name] = self.ax.plot(x, y, 'o', color=color, label=name)[0]
            else:
                self.ax.plot(x, y, 'o', color=color)

        # Display all unique object names in the bottom-left legend
        self.ax.legend(handles=legend_labels.values(), loc='lower left')
        plt.draw()
        plt.pause(0.1)
        self.save_plot()

    def save_plot(self):
        filename = 'slam_plot.png'
        self.fig.savefig(filename)
        print(f"Plot saved as {filename}")

    def spin_and_track(self):
        time.sleep(2)
        for _ in range(4):
            for _ in range(4):
                self.move_forward(0.5)
                self.update_and_plot()
            self.turn_90_degrees()

        self.plot_robot_positions()
        self.save_plot()

    def move_forward(self, distance):
        print("Moving forward by 0.5 meters")
        move_twist = Twist()
        move_twist.linear.x = 2.0
        self.publisher_.publish(move_twist)
        time.sleep(distance / 0.5)
        move_twist.linear.x = 0.0
        self.publisher_.publish(move_twist)

        last_x, last_y = self.ekf_slam.state[0, 0], self.ekf_slam.state[1, 0]
        new_x = last_x + distance * np.cos(self.ekf_slam.state[2, 0])
        new_y = last_y + distance * np.sin(self.ekf_slam.state[2, 0])
        self.robot_positions.append([new_x, new_y])
        print(f"Updated Position: x = {new_x}, y = {new_y}")

    def turn_90_degrees(self):
        print("Turning 90 degrees")
        turn_twist = Twist()
        turn_twist.angular.z = 8.0
        self.publisher_.publish(turn_twist)
        time.sleep(np.pi / 2)
        turn_twist.angular.z = 0.0
        self.publisher_.publish(turn_twist)

        self.ekf_slam.state[2, 0] += np.pi / 2
        self.ekf_slam.state[2, 0] %= 2 * np.pi
        print(f"Updated Heading (theta): {self.ekf_slam.state[2, 0]} radians")

    def plot_robot_positions(self):
        plt.title("Robot Path and Detected Object Positions")
        plt.xlabel("X position (meters)")
        plt.ylabel("Y position (meters)")

def main(args=None):
    rclpy.init(args=args)
    node = SlamControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
