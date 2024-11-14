# ekf_slam_node.py - SLAM Node for EKF SLAM functionality
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

class EKFSLAMNode(Node):
    def __init__(self):
        super().__init__('ekf_slam_node')
        self.subscription = self.create_subscription(
            Float32MultiArray, '/control/detection_info', self.update_callback, 10
        )
        self.control_subscription = self.create_subscription(
            Float32MultiArray, '/control/movement_info', self.predict_callback, 10
        )
        self.robot_state_publisher = self.create_publisher(Float32MultiArray, '/slam/robot_state', 10)
        self.object_state_publisher = self.create_publisher(Float32MultiArray, '/slam/object_states', 10)
        
        # Initialize SLAM state
        self.object_list = ['tv', 'bottle', 'potted plant', 'suitcase', 'umbrella', 'teddy bear', 'backpack', 'stop sign', 'oven', 'airplane']
        self.state = np.zeros((3 + 2 * len(self.object_list), 1))  # [x, y, theta, x1, y1, x2, y2, ...]
        self.P = np.eye(3 + 2 * len(self.object_list)) * 1000  # Large initial uncertainty for landmarks
        self.R = np.diag([0.1, 0.1])  # Measurement noise for range and bearing
        self.Q = np.diag([0.05, 0.05, 0.01])  # Process noise for [x, y, theta]

    def predict_callback(self, msg):
        # Prediction step based on received control input
        distance, heading_change = msg.data
        x, y, theta = self.state[0, 0], self.state[1, 0], self.state[2, 0]

        # Calculate new position based on control input
        new_x = x + distance * np.cos(theta)
        new_y = y + distance * np.sin(theta)
        new_theta = theta + heading_change
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi  # Normalize theta

        # Update state and covariance
        self.state[0, 0], self.state[1, 0], self.state[2, 0] = new_x, new_y, new_theta
        F = np.eye(len(self.state))
        F[0, 2] = -distance * np.sin(theta)
        F[1, 2] = distance * np.cos(theta)
        Q_expanded = np.zeros_like(self.P)
        Q_expanded[:3, :3] = self.Q
        self.P = F @ self.P @ F.T + Q_expanded

        self.publish_state()

    def update_callback(self, msg):
        # Update step based on detected object measurement
        obj_x, obj_y, obj_index = msg.data
        obj_index = int(obj_index)
        landmark_idx = 3 + 2 * obj_index
        x, y, theta = self.state[0, 0], self.state[1, 0], self.state[2, 0]

        # If the object has high uncertainty, initialize it
        if self.P[landmark_idx, landmark_idx] > 999:
            self.state[landmark_idx, 0] = obj_x
            self.state[landmark_idx + 1, 0] = obj_y
            self.P[landmark_idx:landmark_idx + 2, landmark_idx:landmark_idx + 2] = np.eye(2) * 100

        # Perform EKF update with innovation
        delta_x, delta_y = self.state[landmark_idx, 0] - x, self.state[landmark_idx + 1, 0] - y
        q = delta_x**2 + delta_y**2
        predicted_distance, predicted_bearing = np.sqrt(q), np.arctan2(delta_y, delta_x) - theta
        actual_distance, actual_bearing = np.sqrt((obj_x - x)**2 + (obj_y - y)**2), np.arctan2(obj_y - y, obj_x - x) - theta
        innovation = np.array([[actual_distance - predicted_distance], [actual_bearing - predicted_bearing]])
        innovation[1, 0] = (innovation[1, 0] + np.pi) % (2 * np.pi) - np.pi

        # Calculate Jacobian H
        H = np.zeros((2, len(self.state)))
        H[0, 0], H[0, 1] = -delta_x / predicted_distance, -delta_y / predicted_distance
        H[1, 0], H[1, 1] = delta_y / q, -delta_x / q
        H[0, landmark_idx], H[0, landmark_idx + 1] = delta_x / predicted_distance, delta_y / predicted_distance
        H[1, landmark_idx], H[1, landmark_idx + 1] = -delta_y / q, delta_x / q

        # Update EKF state and covariance with Kalman gain
        S, K = H @ self.P @ H.T + self.R, self.P @ H.T @ np.linalg.inv(S)
        self.state += K @ innovation
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P

        self.publish_state()

    def publish_state(self):
        # Publish robot state
        robot_state_msg = Float32MultiArray()
        robot_state_msg.data = [self.state[0, 0], self.state[1, 0], self.state[2, 0]]
        self.robot_state_publisher.publish(robot_state_msg)

        # Publish all object states
        object_state_msg = Float32MultiArray()
        object_states = [self.state[3 + 2 * i, 0] for i in range(len(self.object_list))]
        object_state_msg.data = object_states
        self.object_state_publisher.publish(object_state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = EKFSLAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
