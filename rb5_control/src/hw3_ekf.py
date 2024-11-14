# ekf_slam_node.py - EKF SLAM as a separate ROS node
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

class EKFSLAMNode(Node):
    def __init__(self):
        super().__init__('ekf_slam_node')
        self.declare_parameter('object_list', ['tv', 'bottle', 'potted plant', 'suitcase', 'umbrella', 'teddy bear', 'backpack', 'stop sign', 'oven', 'airplane'])
        self.object_list = self.get_parameter('object_list').value
        self.state = np.zeros((3 + 2 * len(self.object_list), 1))  # [x, y, theta, x1, y1, x2, y2, ...]
        self.P = np.eye(3 + 2 * len(self.object_list)) * 1000  # Large initial uncertainty for landmarks
        self.R = np.diag([0.1, 0.1])  # Measurement noise for range and bearing
        self.Q = np.diag([0.05, 0.05, 0.01])  # Process noise for [x, y, theta]

        self.create_subscription(Float32MultiArray, '/movement_command', self.handle_movement_command, 10)
        self.create_subscription(Float32MultiArray, '/detected_object_info', self.handle_object_detection, 10)
        self.state_publisher = self.create_publisher(Float32MultiArray, '/ekf_slam_state', 10)

    def handle_movement_command(self, msg):
        distance, heading_change = msg.data
        self.predict([distance, heading_change])

    def predict(self, control_input):
        x, y, theta = self.state[0, 0], self.state[1, 0], self.state[2, 0]
        distance, heading_change = control_input

        new_x = x + distance * np.cos(theta)
        new_y = y + distance * np.sin(theta)
        new_theta = (theta + heading_change + np.pi) % (2 * np.pi) - np.pi

        self.state[0, 0], self.state[1, 0], self.state[2, 0] = new_x, new_y, new_theta

        Q_expanded = np.zeros_like(self.P)
        Q_expanded[:3, :3] = self.Q

        F = np.eye(len(self.state))
        F[0, 2], F[1, 2] = -distance * np.sin(theta), distance * np.cos(theta)
        self.P = F @ self.P @ F.T + Q_expanded

        self.publish_state()

    def handle_object_detection(self, msg):
        distance, angle, obj_index = msg.data
        obj_index = int(obj_index)

        x, y, theta = self.state[0, 0], self.state[1, 0], self.state[2, 0]
        obj_x = x + distance * np.cos(theta + angle)
        obj_y = y + distance * np.sin(theta + angle)

        self.update((obj_x, obj_y), obj_index)

    def update(self, measurement, obj_index):
        x, y, theta = self.state[0, 0], self.state[1, 0], self.state[2, 0]
        obj_x, obj_y = measurement
        landmark_idx = 3 + 2 * obj_index

        # Initialize landmark position if it has high uncertainty
        if self.P[landmark_idx, landmark_idx] > 999:
            self.state[landmark_idx, 0], self.state[landmark_idx + 1, 0] = obj_x, obj_y
            self.P[landmark_idx:landmark_idx + 2, landmark_idx:landmark_idx + 2] = np.eye(2) * 100

        # Calculate predicted measurement
        delta_x, delta_y = self.state[landmark_idx, 0] - x, self.state[landmark_idx + 1, 0] - y
        q = delta_x**2 + delta_y**2
        predicted_distance = np.sqrt(q)
        predicted_bearing = np.arctan2(delta_y, delta_x) - theta

        # Actual measurement
        actual_distance = np.sqrt((obj_x - x)**2 + (obj_y - y)**2)
        actual_bearing = np.arctan2(obj_y - y, obj_x - x) - theta
        innovation = np.array([[actual_distance - predicted_distance], [(actual_bearing - predicted_bearing + np.pi) % (2 * np.pi) - np.pi]])

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

        # Compute the innovation covariance S and Kalman gain K
        S = H @ self.P @ H.T + self.R  # S is now computed separately
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update the state and covariance matrix
        self.state += K @ innovation
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P

        # Publish the updated state
        self.publish_state()

    def publish_state(self):
        msg = Float32MultiArray()
        msg.data = self.state.flatten().tolist()
        self.state_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = EKFSLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
