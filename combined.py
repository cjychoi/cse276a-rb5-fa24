import cv2
import numpy as np
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import math
import matplotlib.pyplot as plt
from mpi_control import MegaPiController


class YoloCameraNode(Node):
    def __init__(self):
        super().__init__('yolo_camera_node')

        # Set up parameters
        self.declare_parameter('camera_id', '0')
        self.declare_parameter('topic_name', '/camera_0')
        self.declare_parameter('frame_rate', 30)

        self.camera_id = self.get_parameter('camera_id').value
        self.topic_name = self.get_parameter('topic_name').value
        self.frame_rate = self.get_parameter('frame_rate').value

        self.bridge = CvBridge()
        self.frame = None
        self.target_object = "person"  # Object to detect
        self.model = YOLO('yolov8n.pt')

        # Focal length and known width for distance estimation
        self.KNOWN_WIDTH = 8.0  # Adjust according to the object in cm
        self.FOCAL_LENGTH = 902.8  # Adjust based on your camera calibration

        self.CAMERA_WIDTH = 640  # Width of camera frame in pixels
        self.CAMERA_CENTER = self.CAMERA_WIDTH / 2

        # Instantiate WaypointNavigator
        self.navigator = WaypointNavigator(waypoint_file='waypoints.txt')

        # Matplotlib setup for real-time plotting
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Robot Movement")
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.robot_path_x = []
        self.robot_path_y = []

        # Subscribe to the camera topic
        self.subscription = self.create_subscription(
            Image, self.topic_name, self.image_callback, 10
        )

        # Timer to process frames
        self.timer = self.create_timer(1 / self.frame_rate, self.process_frame)

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting ROS Image to OpenCV: {e}")

    def estimate_distance(self, bbox_width):
        if bbox_width > 0:
            return (self.KNOWN_WIDTH * self.FOCAL_LENGTH) / bbox_width
        return None

    def calculate_angle_to_center(self, bbox_center_x):
        offset = bbox_center_x - self.CAMERA_CENTER
        FOV = 1.0472  # Field of view in radians (example: 60 degrees)
        angle_to_rotate = (offset / self.CAMERA_WIDTH) * FOV
        return angle_to_rotate

    def process_frame(self):
        if self.frame is None:
            self.get_logger().info("No frame received yet...")
            return

        # Run YOLO detection
        results = self.model(self.frame)

        if len(results) == 0 or results[0].boxes.shape[0] == 0:
            self.get_logger().info("Object not found. Rotating -45 degrees.")
            self.rotate_and_search(-45)
            return

        for result in results:
            detection_count = result.boxes.shape[0]
            for i in range(detection_count):
                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]
                if name == self.target_object:
                    self.get_logger().info(f'Object found: {self.target_object}')

                    # Extract bounding box and calculate distance
                    bounding_box = result.boxes.xyxy[i].cpu().numpy()
                    x = int(bounding_box[0])
                    width = int(bounding_box[2] - x)
                    bbox_center_x = x + width / 2
                    distance = self.estimate_distance(width)

                    # Calculate angle to rotate
                    angle_to_rotate = self.calculate_angle_to_center(bbox_center_x)
                    if abs(angle_to_rotate) > 0.1:
                        self.navigator.rotate_to_angle(angle_to_rotate)
                    else:
                        self.navigator.move_straight(distance - 0.1)

                    # Plot the robot's movement
                    self.update_plot()

    def rotate_and_search(self, degrees):
        radians_to_rotate = math.radians(degrees)
        self.navigator.rotate_to_angle(radians_to_rotate)
        self.get_logger().info("Searching for object again after rotation.")

    def update_plot(self):
        x, y, _ = self.navigator.get_current_position()
        self.robot_path_x.append(x)
        self.robot_path_y.append(y)
        self.ax.plot(self.robot_path_x, self.robot_path_y, 'bo-')
        plt.pause(0.001)


class WaypointNavigator:
    def __init__(self, waypoint_file):
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)

        self.waypoints = self.load_waypoints(waypoint_file)
        self.current_waypoint_idx = 0
        self.k_v = 30
        self.k_w = 55
        self.dist_per_sec = 10 / 1
        self.rad_per_sec = math.pi / 2
        self.tolerance = 0.1
        self.current_position = [0.0, 0.0, 0.0]

    def load_waypoints(self, filename):
        waypoints = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                name, known_width = line.strip().split(',')
                waypoints.append((name, float(known_width)))
        return waypoints

    def get_current_position(self):
        return self.current_position

    def set_current_position(self, waypoint):
        self.current_position = waypoint

    def rotate_to_angle(self, angle_diff):
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

    def move_straight(self, distance):
        movement_time = abs(distance) / (self.dist_per_sec / 100)
        self.mpi_ctrl.carStraight(self.k_v)
        time.sleep(movement_time)
        self.mpi_ctrl.carStop()


def main(args=None):
    rclpy.init(args=args)
    yolo_camera_node = YoloCameraNode()
    rclpy.spin(yolo_camera_node)
    yolo_camera_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
