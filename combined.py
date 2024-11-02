import time
import math
import rclpy
from rclpy.node import Node
from mpi_control import MegaPiController
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

current_position = [0.0, 0.0, 0.0]

class WaypointNavigator(Node):
    global current_position

    def __init__(self, object_file):
        super().__init__('navigator')
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)
        self.landmarks = self.load_landmarks(object_file)
        self.current_landmark_idx = 0
        self.k_v = 30
        self.k_w = 55
        self.dist_per_sec = 10 / 1
        self.rad_per_sec = math.pi / 2
        self.tolerance = 0.1
        self.path_x = []
        self.path_y = []

        # Camera setup
        self.bridge = CvBridge()
        self.frame = None
        self.target_object = None

        # YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        self.KNOWN_WIDTH = 8.0
        self.FOCAL_LENGTH = 902.8

        # Subscribe to camera
        self.subscription = self.create_subscription(
            Image,
            '/camera_0',
            self.image_callback,
            10
        )
        self.timer = self.create_timer(1 / 30, self.process_frame)

    def load_landmarks(self, filename):
        landmarks = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                object_name, known_width, final_pose = map(str.strip, line.split(', '))
                known_width = float(known_width)
                final_pose = float(final_pose)  # Assuming this is the final angle for rotation
                landmarks.append((object_name, known_width, final_pose))
        return landmarks

    def image_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting ROS Image to OpenCV: {e}")

    def estimate_distance(self, bbox_width):
        if bbox_width > 0:
            return (self.KNOWN_WIDTH * self.FOCAL_LENGTH) / bbox_width
        return None

    def process_frame(self):
        if self.frame is None:
            return

        results = self.model(self.frame)
        for result in results:
            detection_count = result.boxes.shape[0]
            for i in range(detection_count):
                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]

                if name == self.target_object:
                    bounding_box = result.boxes.xyxy[i].cpu().numpy()
                    width = int(bounding_box[2] - bounding_box[0])

                    # Estimate distance
                    distance = self.estimate_distance(width)
                    if distance:
                        self.get_logger().info(f"Object {self.target_object} found. Distance: {distance} cm")
                        self.move_to_object(distance)

    def move_to_object(self, distance):
        # Move forward distance - 10 cm
        move_distance = distance - 10
        movement_time = move_distance / (self.dist_per_sec / 100)
        self.mpi_ctrl.carStraight(self.k_v)
        time.sleep(movement_time)
        self.mpi_ctrl.carStop()

    def rotate_to_final_pose(self, final_pose):
        # Rotate to the specified final pose after moving to the object
        _, _, current_angle = self.get_current_position()
        angle_diff = final_pose - current_angle
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

    def search_for_object(self):
        # Rotate -45 degrees until object is found
        while True:
            if self.frame is not None:
                results = self.model(self.frame)
                for result in results:
                    detection_count = result.boxes.shape[0]
                    for i in range(detection_count):
                        cls = int(result.boxes.cls[i].item())
                        name = result.names[cls]
                        if name == self.target_object:
                            return

            self.mpi_ctrl.carRotate(-self.k_w)
            time.sleep(math.pi / 4 / self.rad_per_sec)
            self.mpi_ctrl.carStop()

    def start_navigation(self):
        for landmark in self.landmarks:
            object_name, known_width, final_pose = landmark
            self.target_object = object_name

            self.search_for_object()
            self.process_frame()

            self.rotate_to_final_pose(final_pose)
            self.path_x.append(current_position[0])
            self.path_y.append(current_position[1])

        self.plot_path()

    def plot_path(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.path_x, self.path_y, marker='o', label="Robot Path")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Robot Movement Path")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_current_position(self):
        return current_position[0], current_position[1], current_position[2]

    def set_current_position(self, position):
        current_position[0], current_position[1], current_position[2] = position

if __name__ == "__main__":
    rclpy.init(args=None)
    
    navigator = WaypointNavigator(object_file='objects.txt')

    navigator.start_navigation()

    rclpy.spin(navigator)

    navigator.destroy_node()
    rclpy.shutdown()
