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
from rclpy.parameter import Parameter

# Initialize current position (x, y, theta)
current_position = [0.0, 0.0, 0.0]

class WaypointNavigator(Node):
    def __init__(self, landmark_file):
        super().__init__('waypoint_navigator')

        # Initialize robot controller
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)

        # Load landmarks (object name, known size, final pose)
        self.landmarks = self.load_landmarks(landmark_file)

        # Initialize navigation parameters
        self.k_v = 30
        self.k_w = 55
        self.dist_per_sec = 10 / 1  # 10 cm per second
        self.rad_per_sec = math.pi / 2  # pi/2 radians per second
        self.tolerance = 0.1  # Tolerance for centering the object
        self.path_x = []
        self.path_y = []

        # YOLO model setup
        self.model = YOLO('yolov8n.pt')
        self.KNOWN_WIDTH = 8.0  # Example object width in cm
        self.FOCAL_LENGTH = 902.8  # Camera focal length in pixels

        # Set up image processing (camera)
        self.bridge = CvBridge()
        self.frame = None
        self.subscription = self.create_subscription(
            Image, '/camera_0', self.image_callback, 10)

        # For plotting
        self.path_x.append(current_position[0])
        self.path_y.append(current_position[1])

    def image_callback(self, msg):
        """Converts ROS Image message to OpenCV image."""
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting ROS Image to OpenCV: {e}")

    def load_landmarks(self, filename):
        """Loads landmarks from a file."""
        landmarks = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                obj_name, known_width, x_goal, y_goal, theta_goal = line.strip().split(',')
                landmarks.append((obj_name, float(known_width), float(x_goal), float(y_goal), float(theta_goal)))
        return landmarks

    def estimate_distance(self, bbox_width):
        """Estimates the distance to the object based on bounding box width."""
        if bbox_width > 0:
            return (self.KNOWN_WIDTH * self.FOCAL_LENGTH) / bbox_width
        return None

    def rotate_robot(self, angle):
        """Rotates the robot by a given angle in radians."""
        rotation_time = abs(angle) / self.rad_per_sec
        self.mpi_ctrl.carRotate(self.k_w if angle > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

    def move_straight(self, distance):
        """Moves the robot straight by a specified distance."""
        movement_time = abs(distance) / (self.dist_per_sec / 100)
        self.mpi_ctrl.carStraight(self.k_v)
        time.sleep(movement_time)
        self.mpi_ctrl.carStop()

    def search_for_object(self, target_object):
        """Searches for the object using the YOLO model, rotating until found."""
        while True:
            if self.frame is None:
                self.get_logger().info("Waiting for camera frame...")
                time.sleep(0.1)
                continue

            results = self.model(self.frame)

            for result in results:
                detection_count = result.boxes.shape[0]
                for i in range(detection_count):
                    cls = int(result.boxes.cls[i].item())
                    name = result.names[cls]
                    if name == target_object:
                        self.get_logger().info(f'Found object: {target_object}')
                        bounding_box = result.boxes.xyxy[i].cpu().numpy()
                        x = int(bounding_box[0])
                        width = int(bounding_box[2] - x)

                        # Calculate the angle to rotate
                        center_offset = (x + width / 2) - (self.frame.shape[1] / 2)
                        angle_to_rotate = -center_offset / self.frame.shape[1] * math.pi / 2

                        # Rotate until object is centered
                        if abs(center_offset) > self.tolerance:
                            self.rotate_robot(angle_to_rotate)
                        else:
                            distance = self.estimate_distance(width)
                            return distance

            # Rotate -45 degrees if object is not found
            self.get_logger().info(f'Object not found, rotating -45 degrees')
            self.rotate_robot(-math.pi / 4)

    def navigate_to_landmark(self, landmark):
        """Navigates to a specific landmark."""
        obj_name, known_width, x_goal, y_goal, theta_goal = landmark
        self.KNOWN_WIDTH = known_width

        self.get_logger().info(f"Navigating to landmark: {obj_name}")

        # Search for the object using YOLO
        distance = self.search_for_object(obj_name)
        if distance is not None:
            self.get_logger().info(f"Estimated distance to {obj_name}: {distance} cm")

            # Move forward (distance - 10 cm)
            self.move_straight(distance - 10)

            # Rotate to the final orientation
            self.get_logger().info(f"Rotating to final pose for {obj_name}")
            self.rotate_robot(theta_goal)

            # Log and plot path
            self.set_current_position([x_goal, y_goal, theta_goal])
            self.plot_path()

    def set_current_position(self, waypoint):
        """Updates the robot's current position."""
        current_position[0], current_position[1], current_position[2] = waypoint
        self.path_x.append(waypoint[0])
        self.path_y.append(waypoint[1])

    def plot_path(self):
        """Plots the robot's path using matplotlib."""
        plt.figure(figsize=(8, 6))
        plt.plot(self.path_x, self.path_y, marker='o', label="Robot Path")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Robot Movement Path")
        plt.legend()
        plt.grid(True)
        plt.show()

    def start_navigation(self):
        """Starts the navigation process for all landmarks."""
        for landmark in self.landmarks:
            self.navigate_to_landmark(landmark)

        self.get_logger().info("All landmarks visited.")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    # Start the waypoint navigator with landmarks
    navigator = WaypointNavigator(landmark_file='object.txt')
    navigator.start_navigation()

    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
