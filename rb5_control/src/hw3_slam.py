#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
import cv2
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import math
import matplotlib.pyplot as plt

# EKF SLAM Class
class EKFSLAM:
    def __init__(self, objects_to_detect, focal_length, known_width):
        # Initialize the EKF state and covariance
        self.state = np.zeros((3 + 2 * len(objects_to_detect), 1))  # [x, y, theta, x1, y1, x2, y2, ...]
        self.P = np.eye(3 + 2 * len(objects_to_detect)) * 1e-3  # Initial covariance (uncertainty)
        self.objects_to_detect = objects_to_detect  # List of objects to track
        self.focal_length = focal_length  # Focal length of camera
        self.known_width = known_width  # Known width of objects (cm)

    def predict(self, u):
        """ Prediction step of the EKF. """
        # u = [dx, dy, dtheta] (robot's control input)
        theta = self.state[2, 0]
        G = np.eye(len(self.state))  # Jacobian matrix for the prediction
        G[0, 2] = -u[0] * np.sin(theta)
        G[1, 2] = u[0] * np.cos(theta)
        Q = np.eye(len(self.state)) * 1e-3  # Process noise
        self.state[:3] += u  # Update robot's state with control input
        self.P = G @ self.P @ G.T + Q  # Update covariance

    def update(self, z, landmark_index):
        """ Update step of the EKF. """
        # z = [distance, angle] to detected object
        # Update the position of the detected object in the state vector
        H = np.zeros((2, len(self.state)))  # Jacobian for the measurement model
        H[0, 3 + 2 * landmark_index] = 1
        H[1, 3 + 2 * landmark_index + 1] = 1

        R = np.eye(2) * 1e-3  # Measurement noise
        innovation = z - self.state[3 + 2 * landmark_index: 3 + 2 * (landmark_index + 1)]  # Innovation
        S = H @ self.P @ H.T + R  # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain

        # Update state with Kalman gain
        self.state += K @ innovation
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P

    def get_state(self):
        """ Return the current state. """
        return self.state

# YOLO Camera Node with EKF-SLAM Integration
class YoloCameraNode(Node):
    def __init__(self):
        super().__init__('yolo_camera_node')
        self.declare_parameter('camera_id', '0')
        self.declare_parameter('topic_name', '/camera_0')
        self.declare_parameter('frame_rate', 30)
        self.camera_id = self.get_parameter('camera_id').value
        self.topic_name = self.get_parameter('topic_name').value
        self.br = CvBridge()

        # Object list with known width (in meters, for simplicity all objects are 20 cm)
        self.objects_to_detect = {
            'teddy bear': 0.2,
            'bottle': 0.2
        }
        self.current_object_index = 0
        self.detection_timeout = 10  # Timeout after 10 seconds of searching
        self.detection_start_time = time.time()

        # YOLOv8 Model
        self.model = YOLO('yolov8n.pt')

        # Initialize camera feed subscription
        self.subscription = self.create_subscription(
            Image, self.topic_name, self.image_callback, 10
        )
        
        # Set up a publisher to publish distances
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)

        # Initialize EKF
        self.ekf_slam = EKFSLAM(
            list(self.objects_to_detect.keys()),
            focal_length=902.8,  # Approximate focal length
            known_width=0.2  # Known object width (20 cm)
        )

    def image_callback(self, msg):
        """Callback function for processing the camera feed."""
        print("\n<<image_callback>>\n")
        cv_image = self.br.imgmsg_to_cv2(msg, 'bgr8')
        
        # Use YOLO model to detect objects 
        results = self.model(cv_image)
        detected_objects = results[0].boxes

        # Find and track the desired object in sequence
        for box in detected_objects:
            cls = int(box.cls.item())  # Get the class index
            object_name = self.model.names[cls]  # Map class index to object name
            if object_name == list(self.objects_to_detect.keys())[self.current_object_index]:
                print("\n<<Object Found!>>\n")
                self.handle_detected_object(cv_image, box)
                return

        # Timeout mechanism: if object is not found within a given time, rotate to search again
        if time.time() - self.detection_start_time > self.detection_timeout:
            print("\n<<Object Not Found - rotate_to_search>>\n")
            self.rotate_to_search()

    def handle_detected_object(self, cv_image, box):
        """Handle the detected object, calculate distance, and publish twist commands."""
        print("\n<<handle_detected_object>>\n")

        # Extract bounding box center
        x_min, y_min, x_max, y_max = box.xyxy[0]  # Access the bounding box coordinates
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min

        # Estimate distance based on object size or known dimensions
        distance = (self.ekf_slam.known_width * self.ekf_slam.focal_length) / width

        # Log detection
        self.get_logger().info(f'Detected {list(self.objects_to_detect.keys())[self.current_object_index]} at {distance:.2f}m')

        # Move towards the object: Adjust the robot's velocity based on distance and object position
        move_twist = Twist()
        move_twist.linear.x = float(min(distance - 0.1, 0.1))  # Move forward, stop 10 cm before the object
        move_twist.angular.z = float(-(x_center - (cv_image.shape[1] / 2)) / 500)  # Adjust heading
        self.publisher_.publish(move_twist)

        # EKF Update
        self.ekf_slam.update(np.array([distance, x_center]), self.current_object_index)

        # If within a threshold distance, stop and move to the next object
        if distance < 0.2:
            self.current_object_index += 1
            self.detection_start_time = time.time()  # Reset detection timer

    def rotate_to_search(self):
        """Rotate the robot to search for objects."""
        self.get_logger().info('Rotating to search for the next object...')
        rotate_twist = Twist()
        rotate_twist.angular.z = 0.1  # Rotate at a fixed speed
        self.publisher_.publish(rotate_twist)

        # Reset detection start time after rotation
        self.detection_start_time = time.time()

    def plot_world_frame(self):
        """Plot the robot and detected objects in the world frame."""
        state = self.ekf_slam.get_state()
        robot_x, robot_y, _ = state[0, 0], state[1, 0], state[2, 0]

        plt.figure()
        plt.scatter(robot_x, robot_y, c='blue', label='Robot')
        
        for i, obj_name in enumerate(self.objects_to_detect.keys()):
            obj_x = state[3 + 2 * i, 0]
            obj_y = state[3 + 2 * i + 1, 0]
            plt.scatter(obj_x, obj_y, c='red', label=obj_name)

        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.legend()
        plt.title('Robot and Object Locations in World Frame')
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    
    # Initialize YOLO object detector
    yolo_node = YoloCameraNode()

    # Start the robot navigation
    rclpy.spin(yolo_node)

    # After finishing, plot the world frame
    yolo_node.plot_world_frame()
    
    yolo_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
