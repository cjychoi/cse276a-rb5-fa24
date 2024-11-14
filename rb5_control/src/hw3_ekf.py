# ekf_slam_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

class EKFSLAM(Node):
    def __init__(self, object_list):
        super().__init__('ekf_slam_node')
        self.state = np.zeros((3 + 2 * len(object_list), 1))  # [x, y, theta, x1, y1, x2, y2, ...]
        self.P = np.eye(3 + 2 * len(object_list)) * 1000  # Large initial uncertainty for landmarks
        self.R = np.diag([0.1, 0.1])  # Measurement noise for range and bearing
        self.Q = np.diag([0.05, 0.05, 0.01])  # Process noise for [x, y, theta]
        self.object_list = object_list

        # Subscriber to receive movement commands
        self.movement_sub = self.create_subscription(
            Float32MultiArray, '/movement_command', self.movement_callback, 10
        )

        # Subscriber to receive detected objects
        self.object_sub = self.create_subscription(
            Float32MultiArray, '/detected_object_info', self.update_callback, 10
        )

        # Publisher to send updated SLAM state
        self.state_pub = self.create_publisher(Float32MultiArray, '/ekf_slam_state', 10)

    def predict(self, control_input):
        """Predict step for EKF based on control input."""
        x, y, theta = self.state[0, 0], self.state[1, 0], self.state[2, 0]
        distance, heading_change = control_input

        new_x = x + distance * np.cos(theta)
        new_y = y + distance * np.sin(theta)
        new_theta = theta + heading_change
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi

        self.state[0, 0] = new_x
        self.state[1, 0] = new_y
        self.state[2, 0] = new_theta

        Q_expanded = np.zeros_like(self.P)
        Q_expanded[:3, :3] = self.Q

        F = np.eye(len(self.state))
        F[0, 2] = -distance * np.sin(theta)
        F[1, 2] = distance * np.cos(theta)
        self.P = F @ self.P @ F.T + Q_expanded

    def update(self, measurement, obj_index):
        x, y, theta = self.state[0, 0], self.state[1, 0], self.state[2, 0]
        obj_x, obj_y = measurement
        landmark_idx = 3 + 2 * int(obj_index)
        
        if self.P[landmark_idx, landmark_idx] > 999:
            self.state[landmark_idx, 0] = obj_x
            self.state[landmark_idx + 1, 0] = obj_y
            self.P[landmark_idx:landmark_idx + 2, landmark_idx:landmark_idx + 2] = np.eye(2) * 100

        delta_x = self.state[landmark_idx, 0] - x
        delta_y = self.state[landmark_idx + 1, 0] - y
        q = delta_x**2 + delta_y**2
        predicted_distance = np.sqrt(q)
        predicted_bearing = np.arctan2(delta_y, delta_x) - theta

        actual_distance = np.sqrt((obj_x - x)**2 + (obj_y - y)**2)
        actual_bearing = np.arctan2(obj_y - y, obj_x - x) - theta
        innovation = np.array([[actual_distance - predicted_distance], [actual_bearing - predicted_bearing]])
        innovation[1, 0] = (innovation[1, 0] + np.pi) % (2 * np.pi) - np.pi

        H = np.zeros((2, len(self.state)))
        H[0, 0] = -delta_x / predicted_distance
        H[0, 1] = -delta_y / predicted_distance
        H[1, 0] = delta_y / q
        H[1, 1] = -delta_x / q
        H[0, landmark_idx] = delta_x / predicted_distance
        H[0, landmark_idx + 1] = delta_y / predicted_distance
        H[1, landmark_idx] = -delta_y / q
        H[1, landmark_idx + 1] = delta_x / q

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state += K @ innovation
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P

    def movement_callback(self, msg):
        distance, angle, obj_index = msg.data[:3]  # Extract only the distance and angle for control input
        self.predict([distance, angle])
        self.publish_slam_state()

    def update_callback(self, msg):
        obj_x, obj_y, obj_index = msg.data
        self.update([obj_x, obj_y], int(obj_index))
        self.publish_slam_state()

    def publish_slam_state(self):
        state_msg = Float32MultiArray()
        state_msg.data = np.concatenate((self.state[:3].flatten(), self.state[3:].flatten())).tolist()
        self.state_pub.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = EKFSLAM(object_list=['tv', 'bottle', 'potted plant', 'suitcase', 'umbrella', 'teddy bear', 'backpack', 'stop sign', 'oven', 'airplane'])
    print("EKF")
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