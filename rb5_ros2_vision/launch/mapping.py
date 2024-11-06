import numpy as np
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

class EKFSLAM(Node):
    def __init__(self):
        super().__init__('ekf_slam')

        # Robot state: [x, y, theta] (position and orientation)
        self.state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        
        # State covariance matrix
        self.P = np.eye(3) * 1e-3

        # Process noise covariance (for motion model)
        self.Q = np.diag([0.1, 0.1, np.deg2rad(5)]) ** 2

        # Measurement noise covariance (for landmarks)
        self.R = np.diag([0.2, np.deg2rad(2)]) ** 2

        # Landmarks: Map of landmarks and their estimated positions
        self.landmarks = {}
        self.landmark_covariances = {}

        # EKF Parameters
        self.motion_noise = 0.1  # Motion model noise
        self.measurement_noise = 0.05  # Measurement noise

        # Subscribe to distance, angle, and movement completion
        self.rotation_sub = self.create_subscription(Float32, '/rotation_angle', self.handle_rotation, 10)
        self.move_sub = self.create_subscription(Float32, '/move_distance', self.handle_movement, 10)
        self.is_moving_sub = self.create_subscription(Bool, '/is_moving', self.is_moving_callback, 10)
        self.camera_sub = self.create_subscription(Image, '/camera_0', self.image_callback, 10)

        # Initialize the YOLO model for object detection
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()

        # Plotting setup for robot trajectory and landmarks
        self.x_history = [self.state[0]]
        self.y_history = [self.state[1]]
        self.landmark_history = {}

        self.fig, self.ax = plt.subplots()
        self.plot_robot_movement()

    def is_moving_callback(self, msg):
        self.is_moving = msg.data

    def handle_rotation(self, msg):
        # EKF prediction step with robot rotation
        theta = msg.data
        self.ekf_predict(0, theta)  # No linear movement, just rotation
        self.plot_robot_movement()

    def handle_movement(self, msg):
        # EKF prediction step with robot movement
        distance = msg.data
        self.ekf_predict(distance, 0)  # No rotation, just forward movement
        self.plot_robot_movement()

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(frame)

        for result in results:
            for i in range(result.boxes.shape[0]):
                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]
                
                # Get bounding box and estimate distance/angle
                bbox = result.boxes.xyxy[i].cpu().numpy()
                x, width = int(bbox[0]), int(bbox[2] - bbox[0])
                bbox_center_x = x + width / 2

                # Estimate distance and angle to the object
                distance = self.estimate_distance(width)
                angle_to_rotate = self.calculate_angle_to_center(bbox_center_x)

                # Perform EKF update step for detected landmark
                self.ekf_update(name, distance, angle_to_rotate)

                self.plot_robot_movement()

    def estimate_distance(self, bbox_width):
        # Assume a known width and focal length for simplicity
        KNOWN_WIDTH = 20  # Example known width of object in cm
        FOCAL_LENGTH = 900  # Example focal length in pixels
        return (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width

    def calculate_angle_to_center(self, bbox_center_x):
        CAMERA_WIDTH = 640
        FOV = 1.0472  # 60 degrees in radians
        offset = bbox_center_x - (CAMERA_WIDTH / 2)
        return (offset / CAMERA_WIDTH) * FOV

    def ekf_predict(self, distance, theta):
        # Robot motion model prediction step
        x, y, theta_robot = self.state

        # Update robot state
        if distance != 0:
            dx = distance * math.cos(theta_robot)
            dy = distance * math.sin(theta_robot)
            self.state[0] += dx
            self.state[1] += dy
        if theta != 0:
            self.state[2] += theta

        # Jacobian of motion model (for linearizing EKF)
        F = np.eye(3)
        if distance != 0:
            F[0, 2] = -distance * math.sin(theta_robot)
            F[1, 2] = distance * math.cos(theta_robot)

        # Update covariance matrix with process noise
        self.P = F @ self.P @ F.T + self.Q

    def ekf_update(self, landmark_name, distance, angle):
        # EKF update step for the detected landmark
        if landmark_name not in self.landmarks:
            # Initialize landmark position if not already in the map
            lx = self.state[0] + distance * math.cos(self.state[2] + angle)
            ly = self.state[1] + distance * math.sin(self.state[2] + angle)
            self.landmarks[landmark_name] = np.array([lx, ly])
            self.landmark_covariances[landmark_name] = np.eye(2) * 1e-3

        lx, ly = self.landmarks[landmark_name]

        # Measurement prediction
        delta_x = lx - self.state[0]
        delta_y = ly - self.state[1]
        q = delta_x ** 2 + delta_y ** 2
        z_pred = np.array([math.sqrt(q), math.atan2(delta_y, delta_x) - self.state[2]])

        # Jacobian of the measurement model
        H = np.zeros((2, len(self.state)))
        H[0, 0] = -delta_x / math.sqrt(q)
        H[0, 1] = -delta_y / math.sqrt(q)
        H[1, 0] = delta_y / q
        H[1, 1] = -delta_x / q
        H[1, 2] = -1

        # Kalman Gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Measurement residual
        z = np.array([distance, angle])
        y = z - z_pred

        # State update
        self.state += K @ y

        # Covariance update
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P

        # Update landmark position
        self.landmarks[landmark_name] = [lx, ly]

    def plot_robot_movement(self):
        # Clear the plot
        self.ax.clear()

        # Plot the robot's trajectory
        self.ax.plot(self.x_history, self.y_history, marker='o', label='Robot Trajectory')

        # Plot landmarks
        for name, pos in self.landmarks.items():
            lx, ly = pos
            self.ax.plot(lx, ly, 'rx', label=f'Landmark: {name}')

        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('Robot Trajectory and Landmark Map')
        plt.pause(0.1)

def main(args=None):
    rclpy.init(args=args)
    ekf_slam = EKFSLAM()
    rclpy.spin(ekf_slam)

    ekf_slam.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
