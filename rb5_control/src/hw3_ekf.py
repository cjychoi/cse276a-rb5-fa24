# hw3_ekf.py - Adjusted to calculate object positions relative to the robot's current position and update EKF SLAM
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import matplotlib.pyplot as plt
import time

class EKFSLAM(Node):
    def __init__(self, object_list):
        # Initialize SLAM state (robot pose and landmarks)
        super().__init__('ekf_slam_node')
        self.state = np.zeros((3 + 2 * len(object_list), 1))  # [x, y, theta, x1, y1, x2, y2, ...]
        self.P = np.eye(3 + 2 * len(object_list)) * 1000  # Large initial uncertainty for landmarks
        self.R = np.diag([0.1, 0.1])  # Measurement noise for range and bearing
        self.Q = np.diag([0.05, 0.05, 0.01])  # Process noise for [x, y, theta]
        self.object_list = object_list
        self.colors = plt.cm.get_cmap('tab10', len(object_list))

        # self.colors_pub = self.create_publisher(StringArray, '/ekf_slam_colors', 10)
        # publish_slam_colors()
        

        # Subscriber to receive movement commands
        self.movement_sub = self.create_subscription(
            Float32MultiArray, '/movement_command', self.movement_callback, 10
        )

        # Subscriber to receive detected objects
        self.object_sub = self.create_subscription(
            Float32MultiArray, '/detected_object_info', self.update_callback, 10
        )

        # Subscriber to receive update data
        self.update_sub = self.create_subscription(
            Float32MultiArray, '/ekf_update', self.update_callback, 10
        )

        # # Subscriber to receive predict data
        # self.predict_sub = self.create_subscription(
        #     Float32MultiArray, '/ekf_predict', self.predict, 10
        # )

        # Publisher to send updated SLAM state
        self.state_pub = self.create_publisher(Float32MultiArray, '/ekf_slam_state', 10)
        

    def predict(self, msg):
        """Predict step for EKF based on control input."""
        x, y, theta = self.state[0, 0], self.state[1, 0], self.state[2, 0]
        distance, heading_change = msg

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

        print('predict state: ', self.state)
        # self.publish_slam_state()
        state_msg = Float32MultiArray()
        state_msg.data = np.concatenate((self.state[:3].flatten(), self.state[3:].flatten())).tolist()
        self.state_pub.publish(state_msg)

    def update(self, measurement, obj_index):
        """Update step for EKF using the landmark position relative to the robot's original pose."""
        print('starting update - print')
        x, y, theta = self.state[0, 0], self.state[1, 0], self.state[2, 0]
        distance, angle = measurement  # Distance and angle relative to the robot

        # Calculate landmark's position in the world frame (robot's original pose)
        world_x = x + distance * np.cos(theta + angle)
        world_y = y + distance * np.sin(theta + angle)

        # Update state with the calculated landmark position
        print('found object index is : ', obj_index)
        landmark_idx = 3 + 2 * int(obj_index)
        self.state[landmark_idx, 0] = world_x
        self.state[landmark_idx + 1, 0] = world_y
        print('object found at ', self.state[landmark_idx, 0], ', ', self.state[landmark_idx + 1, 0])

        # Initialize landmarks if needed (large initial uncertainty)
        if self.P[landmark_idx, landmark_idx] > 999:
            self.P[landmark_idx:landmark_idx + 2, landmark_idx:landmark_idx + 2] = np.eye(2) * 100

        # Compute measurement prediction
        delta_x = world_x - x
        delta_y = world_y - y
        q = delta_x**2 + delta_y**2
        predicted_distance = np.sqrt(q)
        predicted_bearing = np.arctan2(delta_y, delta_x) - theta

        # Compute the actual measurement
        actual_distance = distance
        actual_bearing = angle
        innovation = np.array([[actual_distance - predicted_distance], [actual_bearing - predicted_bearing]])

        # Normalize bearing
        innovation[1, 0] = (innovation[1, 0] + np.pi) % (2 * np.pi) - np.pi

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

        self.state[0, 0] = x
        self.state[1, 0] = y
        self.state[2, 0] = theta
        print('update state: ', self.state)
        # self.publish_slam_state()
        state_msg = Float32MultiArray()
        state_msg.data = np.concatenate((self.state[:3].flatten(), self.state[3:].flatten())).tolist()
        print("\n ekf state msg:")
        print(state_msg.data)
        self.state_pub.publish(state_msg)
        
    

    # def update(self, measurement, obj_index):
    #     """Update step for EKF using the landmark position relative to world frame."""        
    #     x, y, theta = self.state[0, 0], self.state[1, 0], self.state[2, 0]
    #     obj_x, obj_y = measurement  # World coordinates of the detected object
    #     landmark_idx = 3 + 2 * int(obj_index)
        
    #     if self.P[landmark_idx, landmark_idx] > 999:
    #         self.state[landmark_idx, 0] = obj_x
    #         self.state[landmark_idx + 1, 0] = obj_y
    #         self.P[landmark_idx:landmark_idx + 2, landmark_idx:landmark_idx + 2] = np.eye(2) * 100

    #     # Compute measurement prediction
    #     delta_x = self.state[landmark_idx, 0] - x
    #     delta_y = self.state[landmark_idx + 1, 0] - y
    #     q = delta_x**2 + delta_y**2
    #     predicted_distance = np.sqrt(q)
    #     predicted_bearing = np.arctan2(delta_y, delta_x) - theta

    #     actual_distance = np.sqrt((obj_x - x)**2 + (obj_y - y)**2)
    #     actual_bearing = np.arctan2(obj_y - y, obj_x - x) - theta
    #     innovation = np.array([[actual_distance - predicted_distance], [actual_bearing - predicted_bearing]])
    #     innovation[1, 0] = (innovation[1, 0] + np.pi) % (2 * np.pi) - np.pi  # Normalize bearing

    #     # Calculate Jacobian H of the measurement function
    #     H = np.zeros((2, len(self.state)))
    #     H[0, 0] = -delta_x / predicted_distance
    #     H[0, 1] = -delta_y / predicted_distance
    #     H[1, 0] = delta_y / q
    #     H[1, 1] = -delta_x / q
    #     H[0, landmark_idx] = delta_x / predicted_distance
    #     H[0, landmark_idx + 1] = delta_y / predicted_distance
    #     H[1, landmark_idx] = -delta_y / q
    #     H[1, landmark_idx + 1] = delta_x / q

    #     # Compute the innovation covariance
    #     S = H @ self.P @ H.T + self.R

    #     # Compute the Kalman gain
    #     K = self.P @ H.T @ np.linalg.inv(S)

    #     # Update the state and covariance matrix
    #     self.state += K @ innovation
    #     self.P = (np.eye(len(self.state)) - K @ H) @ self.P

    #     print('update state: ', self.state)
    #     # self.publish_slam_state()
    #     state_msg = Float32MultiArray()
    #     state_msg.data = np.concatenate((self.state[:3].flatten(), self.state[3:].flatten())).tolist()
    #     self.state_pub.publish(state_msg)

    def movement_callback(self, msg):
        distance, angle = msg.data
        print([distance, angle])
        self.predict([distance, angle])

    def update_callback(self, msg):
        obj_x, obj_y, obj_index = msg.data
        self.update([obj_x, obj_y], int(obj_index)-1)

    # def publish_slam_state(self):
    #     state_msg = Float32MultiArray()
    #     state_msg.data = np.concatenate((self.state[:3].flatten(), self.state[3:].flatten())).tolist()
    #     self.state_pub.publish(state_msg)

    # def publish_slam_colors(self):
    #     state_msg = StringArray()
    #     state_msg.data = self.colors
    #     self.colors_pub.publish(state_msg)
    
def main(args=None):
    rclpy.init(args=args)
    node = EKFSLAM(object_list=['tv', 'bottle', 'potted plant', 'suitcase', 'umbrella', 'teddy bear', 'backpack', 'stop sign', 'oven'])
    print("EKF running...")
    try:
        rclpy.spin(node)
        print("EKF spin")
    except KeyboardInterrupt:
        pass
    print("EKF PASS")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
