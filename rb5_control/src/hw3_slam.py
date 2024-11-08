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
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# EKF SLAM Class
class EKFSLAM:
    def __init__(self, objects_to_detect, focal_length, known_width):
        self.state = np.zeros((3 + 2 * len(objects_to_detect), 1))  # [x, y, theta, x1, y1, x2, y2, ...]
        self.P = np.eye(3 + 2 * len(objects_to_detect)) * 1e-3  # Initial covariance
        self.objects_to_detect = objects_to_detect
        self.focal_length = focal_length
        self.known_width = known_width

    def predict(self, u):
        theta = self.state[2, 0]
        G = np.eye(len(self.state))  
        G[0, 2] = -u[0] * np.sin(theta)
        G[1, 2] = u[0] * np.cos(theta)
        Q = np.eye(len(self.state)) * 1e-3  
        self.state[:3] += u  
        self.P = G @ self.P @ G.T + Q

    def update(self, z, landmark_index):
        H = np.zeros((2, len(self.state))) 
        H[0, 3 + 2 * landmark_index] = 1
        H[1, 3 + 2 * landmark_index + 1] = 1
        R = np.eye(2) * 1e-3  
        
        # Ensure `z` and `self.state` slice are compatible
        state_slice = self.state[3 + 2 * landmark_index: 3 + 2 * (landmark_index + 1)]
        if state_slice.shape == (2, 1):
            innovation = (z.reshape(2, 1) - state_slice)
        else:
            raise ValueError(f"Unexpected state slice shape: {state_slice.shape}")

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state with calculated innovation
        self.state += K @ innovation
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P

    def get_state(self):
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

        self.objects_to_detect = {
            'teddy bear': 0.2,
            'bottle': 0.2
        }
        self.current_object_index = 0
        self.detection_timeout = 10
        self.detection_start_time = time.time()

        self.model = YOLO('yolov8n.pt')

        self.subscription = self.create_subscription(
            Image, self.topic_name, self.image_callback, 10
        )
        
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)

        # Initialize EKF
        self.ekf_slam = EKFSLAM(
            list(self.objects_to_detect.keys()),
            focal_length=902.8,
            known_width=0.2
        )

        # Initialize plot for real-time visualization
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-5, 5)  # Adjust limits based on environment size
        self.ax.set_ylim(-5, 5)
        self.robot_line, = self.ax.plot([], [], 'bo', label="Robot")
        self.object_lines = [self.ax.plot([], [], 'ro', label=obj)[0] for obj in self.objects_to_detect.keys()]
        self.robot_data = np.array([0, 0])
        self.object_data = {obj: np.array([0, 0]) for obj in self.objects_to_detect.keys()}

        # Animation update function
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=200, blit=True)

    def image_callback(self, msg):
        print("\n<<image_callback>>\n")
        cv_image = self.br.imgmsg_to_cv2(msg, 'bgr8')
        
        results = self.model(cv_image)
        detected_objects = results[0].boxes

        for box in detected_objects:
            cls = int(box.cls.item())
            object_name = self.model.names[cls]
            if object_name == list(self.objects_to_detect.keys())[self.current_object_index]:
                # Object found
                print("\n<<Object Found!>>\n")
                self.handle_detected_object(cv_image, box)
                return

        if time.time() - self.detection_start_time > self.detection_timeout:
            # Object not found
            print("\n<<Object Not Found - rotate_to_search>>\n")
            self.rotate_to_search()

    def handle_detected_object(self, cv_image, box):
        print("\n<<handle_detected_object>>\n")
        x_min, y_min, x_max, y_max = box.xyxy[0]
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min

        distance = (self.ekf_slam.known_width * self.ekf_slam.focal_length) / width
        self.get_logger().info(f'Detected {list(self.objects_to_detect.keys())[self.current_object_index]} at {distance:.2f}m')

        move_twist = Twist()
        move_twist.linear.x = float(min(distance - 0.1, 0.1))  
        move_twist.angular.z = float(-(x_center - (cv_image.shape[1] / 2)) / 500)
        self.publisher_.publish(move_twist)

        self.ekf_slam.update(np.array([distance, x_center]), self.current_object_index)

        if distance < 0.2:
            self.current_object_index += 1
            self.detection_start_time = time.time()

    def rotate_to_search(self):
        rotate_twist = Twist()
        rotate_twist.angular.z = 0.1  
        self.publisher_.publish(rotate_twist)
        self.detection_start_time = time.time()

    def update_plot(self, frame):
        state = self.ekf_slam.get_state()
        robot_x, robot_y, _ = state[0, 0], state[1, 0], state[2, 0]
        self.robot_data = np.array([robot_x, robot_y])

        self.robot_line.set_data(self.robot_data[0], self.robot_data[1])

        for i, obj_name in enumerate(self.objects_to_detect.keys()):
            obj_x = state[3 + 2 * i, 0]
            obj_y = state[3 + 2 * i + 1, 0]
            self.object_data[obj_name] = np.array([obj_x, obj_y])
            self.object_lines[i].set_data(self.object_data[obj_name][0], self.object_data[obj_name][1])

        return [self.robot_line] + self.object_lines

def main(args=None):
    rclpy.init(args=args)
    
    yolo_node = YoloCameraNode()

    plt.show()
    rclpy.spin(yolo_node)

    yolo_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
