# hw3_slam_control.py - Adjusted to calculate object positions relative to the robot's current position and update EKF SLAM
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
        self.Q = np.diag([0.05, 0.05, 0.01])  # Process noise for [x, y, theta]
        self.object_list = object_list
        self.colors = plt.cm.get_cmap('tab10', len(object_list))

    def predict(self, control_input):
        """Predict step for EKF based on control input."""
        x, y, theta = self.state[0, 0], self.state[1, 0], self.state[2, 0]
        distance, heading_change = control_input

        # Predict the new state based on control input
        new_x = x + distance * np.cos(theta)
        new_y = y + distance * np.sin(theta)
        new_theta = theta + heading_change
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi  # Normalize theta

        # Update state
        self.state[0, 0] = new_x
        self.state[1, 0] = new_y
        self.state[2, 0] = new_theta

        # Create an expanded process noise matrix that matches the size of P
        Q_expanded = np.zeros_like(self.P)
        Q_expanded[:3, :3] = self.Q  # Place Q in the top-left corner to affect only robot pose

        # Update the covariance (process model) with the expanded process noise
        F = np.eye(len(self.state))
        F[0, 2] = -distance * np.sin(theta)
        F[1, 2] = distance * np.cos(theta)
        self.P = F @ self.P @ F.T + Q_expanded

    def update(self, measurement, obj_index):
        """Update step for EKF using the landmark position relative to world frame."""
        x, y, theta = self.state[0, 0], self.state[1, 0], self.state[2, 0]
        obj_x, obj_y = measurement  # World coordinates of the detected object
        landmark_idx = 3 + 2 * int(obj_index)
        
        if self.P[landmark_idx, landmark_idx] > 999:
            self.state[landmark_idx, 0] = obj_x
            self.state[landmark_idx + 1, 0] = obj_y
            self.P[landmark_idx:landmark_idx + 2, landmark_idx:landmark_idx + 2] = np.eye(2) * 100

        # Compute measurement prediction
        delta_x = self.state[landmark_idx, 0] - x
        delta_y = self.state[landmark_idx + 1, 0] - y
        q = delta_x**2 + delta_y**2
        predicted_distance = np.sqrt(q)
        predicted_bearing = np.arctan2(delta_y, delta_x) - theta

        actual_distance = np.sqrt((obj_x - x)**2 + (obj_y - y)**2)
        actual_bearing = np.arctan2(obj_y - y, obj_x - x) - theta
        innovation = np.array([[actual_distance - predicted_distance], [actual_bearing - predicted_bearing]])
        innovation[1, 0] = (innovation[1, 0] + np.pi) % (2 * np.pi) - np.pi  # Normalize bearing

        # Calculate Jacobian H of the measurement function
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

class SlamControlNode(Node):
    def __init__(self):
        super().__init__('slam_control_node')
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)
        self.subscription = self.create_subscription(
            Float32MultiArray, '/detected_object_info', self.object_callback, 10
        )
        self.objects_to_detect = ['tv', 'bottle', 'potted plant', 'suitcase', 'umbrella', 'teddy bear', 'backpack', 'stop sign', 'oven', 'airplane']
        self.ekf_slam = EKFSLAM(self.objects_to_detect)
        self.fig, self.ax = plt.subplots()
        self.set_plot_limits()
        self.robot_positions = []  # Store estimated robot positions from EKF
        self.detected_objects = []  # Store positions of detected objects
  #      self.spin_and_track()

    def set_plot_limits(self):
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)

    def object_callback(self, msg):
        distance, angle, obj_index = msg.data
        robot_x, robot_y, theta = self.ekf_slam.state[0, 0], self.ekf_slam.state[1, 0], self.ekf_slam.state[2, 0]
        obj_x = robot_x + distance * np.cos(theta + angle)
        obj_y = robot_y + distance * np.sin(theta + angle)

        # Update EKF with the world-frame coordinates of the detected object
        self.ekf_slam.update((obj_x, obj_y), int(obj_index))

        object_name = self.objects_to_detect[int(obj_index)]
        print(f"Robot Position: (x={robot_x:.2f}, y={robot_y:.2f}, theta={theta:.2f})")
        print(f"Detected {object_name} at world position (x={obj_x:.2f}, y={obj_y:.2f})")

        self.robot_positions.append([robot_x, robot_y])
        self.detected_objects.append((obj_x, obj_y, object_name))
        self.update_and_plot()

    def update_and_plot(self):
        self.ax.clear()
        self.set_plot_limits()
        self.ax.plot(*zip(*self.robot_positions), 'bo-', label="Robot Path")

        legend_labels = {"Robot Path": self.ax.plot([], [], 'bo-', label="Robot Path")[0]}
        for x, y, name in self.detected_objects:
            color = self.ekf_slam.colors(self.objects_to_detect.index(name))
            if name not in legend_labels:
                legend_labels[name] = self.ax.plot(x, y, 'o', color=color, label=name)[0]
            else:
                self.ax.plot(x, y, 'o', color=color)

        self.ax.legend(handles=legend_labels.values(), loc='lower left')
        plt.draw()
        plt.pause(0.1)
        self.save_plot()

    def save_plot(self):
        filename = 'slam_plot.png'
        self.fig.savefig(filename)
        print(f"Plot saved as {filename}")

    def spin_and_track(self, type, length):
        
        # self.move_forward(0.5)
        # self.save_plot()
        # time.sleep(1)

        # self.move_forward(0.5)
        # self.save_plot()
        # time.sleep(1)
        
        # for _ in range(4):
        #     for _ in range(4):  # Stop every 0.5 meters
        #         self.move_forward(0.5)
        #         self.save_plot()
        #         time.sleep(1)
        #     self.turn_90_degrees()
        #     self.save_plot()
        #     time.sleep(1)

        if (type == 'move'):
            self.move_forward(length)
        elif (type == 'spin'):
            if (length == 90):
                self.turn_90_degrees()
            elif (length == 45):
                self.turn_90_degrees()            # MAKE NEW FUNCTION FOR 45 DEGREE TURN FOR OCTOGON

        self.save_plot()
        time.sleep(1)

        self.plot_final_landmarks()
        self.print_final_coordinates()

    def move_forward(self, distance):
        print("Moving forward by 0.5 meters")
        control_input = [distance, 0]
        self.ekf_slam.predict(control_input)

        move_twist = Twist()
        move_twist.linear.x = 2.0
        self.publisher_.publish(move_twist)
        time.sleep(distance / 0.5)
        move_twist.linear.x = 0.0
        self.publisher_.publish(move_twist)

        robot_x, robot_y = self.ekf_slam.state[0, 0], self.ekf_slam.state[1, 0]
        self.robot_positions.append([robot_x, robot_y])
        print(f"Updated Position: x = {robot_x}, y = {robot_y}")

    def turn_90_degrees(self):
        print("Turning 90 degrees")
        control_input = [0, np.pi / 2]
        self.ekf_slam.predict(control_input)

        turn_twist = Twist()
        turn_twist.angular.z = 8.0
        self.publisher_.publish(turn_twist)
        time.sleep(np.pi / 2)
        turn_twist.angular.z = 0.0
        self.publisher_.publish(turn_twist)

        print(f"Updated Heading (theta): {self.ekf_slam.state[2, 0]} radians")

    def plot_final_landmarks(self):
        self.ax.clear()
        self.set_plot_limits()
        self.ax.plot(*zip(*self.robot_positions), 'bo-', label="Robot Path")

        legend_labels = {"Robot Path": self.ax.plot([], [], 'bo-', label="Robot Path")[0]}
        for x, y, name in self.detected_objects:
            color = self.ekf_slam.colors(self.objects_to_detect.index(name))
            legend_labels[name] = self.ax.plot(x, y, 'o', color=color, label=name)[0]

        self.ax.legend(handles=legend_labels.values(), loc='lower left')
        plt.title("Robot Path and Final Detected Object Positions")
        plt.xlabel("X position (meters)")
        plt.ylabel("Y position (meters)")
        plt.show()
        self.save_plot()

    def print_final_coordinates(self):
        print("\nFinal Coordinates of Detected Objects:")
        for i, obj_name in enumerate(self.objects_to_detect):
            landmark_idx = 3 + 2 * i
            obj_x, obj_y = self.ekf_slam.state[landmark_idx, 0], self.ekf_slam.state[landmark_idx + 1, 0]
            print(f"{obj_name}: (x = {obj_x:.2f}, y = {obj_y:.2f})")

def main(args=None):
    rclpy.init(args=args)
    node = SlamControlNode()

    print("SLAM 1")

    # TRY 1
    for _ in range(4):
        for _ in range(4):  # Stop every 0.5 meters
            print("SLAM loop")
            node.spin_and_track('move', 0.5)
            time.sleep(1)
        node.spin_and_track('spin', 90)
        time.sleep(1)
        
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


    # # TRY 2
    # try:
    #     rclpy.spin(node)
    # except KeyboardInterrupt:
    #     pass

    # for _ in range(4):
    #     for _ in range(4):  # Stop every 0.5 meters
    #         node.spin_and_track('move', 0.5)
    #         time.sleep(1)
    #     node.spin_and_track('spin', 90)
    #     time.sleep(1)


    # # TRY 3
    # for _ in range(4):
    #     for _ in range(4):  # Stop every 0.5 meters
    #         node.spin_and_track('move', 0.5)
    #         time.sleep(1)
    #         # rclpy.spin_once(node)
    #         # time.sleep(1)
    #     node.spin_and_track('spin', 90)
    #     time.sleep(1)
    #     # rclpy.spin_once(node)
    #     # time.sleep(1)


    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
