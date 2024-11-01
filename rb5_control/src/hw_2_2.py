import time
import math
import cv2
import numpy as np
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from control import WaypointNavigator
from mpi_control import MegaPiController

class YoloWaypointNavigator(Node):
    def __init__(self):
        super().__init__('yolo_waypoint_navigator')

        # Initialize the MegaPiController for robot control
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)  # Allow some time for the connection to establish

        # Load waypoints from file (these will be object names instead of coordinates)
        self.waypoints = self.load_waypoints('waypoints.txt')
        self.current_waypoint_idx = 0

        # Target object (to be updated dynamically based on waypoints)
        self.target_object = None

        # Control parameters
        self.k_v = 30  # Speed for straight movement
        self.k_w = 55  # Speed for rotational movement
        self.dist_per_sec = 10 / 1  # 10 cm per 1 second at speed 30 for straight movement   
        self.rad_per_sec = math.pi / 2  # Pi radians per 2 seconds at speed 55 for rotational movement
        self.tolerance = 0.1  # Distance tolerance to waypoint (meters)

        # YOLO model setup
        self.model = YOLO('yolov8n.pt')  # YOLOv8 nano model
        self.bridge = CvBridge()

        # Camera parameters for distance estimation
        self.KNOWN_WIDTH = 8.0  # Width of the object in cm (adjust based on the object)
        self.FOCAL_LENGTH = 902.8  # Focal length of the camera in pixels (example)
        self.CAMERA_WIDTH = 640  # Width of the camera frame in pixels
        self.CAMERA_CENTER = self.CAMERA_WIDTH / 2  # Calculate the center of the camera's field of view

        # Subscribe to the camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera_0',
            self.image_callback,
            10
        )

        # Timer to process the camera feed
        self.timer = self.create_timer(1 / 30, self.process_frame)

        # Waypoint handling
        self.current_position = [0.0, 0.0, 0.0]  # x, y, theta
        self.target_object = self.waypoints[self.current_waypoint_idx] if self.waypoints else None

    def load_waypoints(self, filename):
        waypoints = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                waypoints.append(line.strip())  # Object names (like 'person', 'bottle', etc.)
        print('Loaded waypoints (objects):', waypoints)
        return waypoints

    def image_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting ROS Image to OpenCV: {e}")

    def estimate_distance(self, bbox_width):
        if bbox_width > 0:
            distance = (self.KNOWN_WIDTH * self.FOCAL_LENGTH) / bbox_width
            return distance
        else:
            return None

    def calculate_angle_to_center(self, bbox_center_x):
        offset = bbox_center_x - self.CAMERA_CENTER
        FOV = 1.0472  # Field of view of the camera in radians (example: 60 degrees -> 1.0472 radians)
        angle_to_rotate = (offset / self.CAMERA_WIDTH) * FOV
        return angle_to_rotate

    def rotate_to_angle(self, angle_diff):
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

    def move_straight(self, distance):
        movement_time = abs(distance) / (self.dist_per_sec / 100)  # Convert cm/s to m/s
        self.mpi_ctrl.carStraight(self.k_v)
        time.sleep(movement_time)
        self.mpi_ctrl.carStop()

    def process_frame(self):
        if self.frame is None or not self.target_object:
            return

        results = self.model(self.frame)

        if len(results) == 0 or results[0].boxes.shape[0] == 0:
            self.get_logger().info(f"Object '{self.target_object}' not found. Rotating -45 degrees.")
            self.rotate_to_angle(-math.pi / 4)  # Rotate -45 degrees
            return

        # Process detections
        for result in results:
            for i in range(result.boxes.shape[0]):
                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]

                if name == self.target_object:
                    self.get_logger().info(f"Object '{self.target_object}' found.")
                    bounding_box = result.boxes.xyxy[i].cpu().numpy()
                    x = int(bounding_box[0])
                    width = int(bounding_box[2] - x)

                    distance = self.estimate_distance(width)
                    bbox_center_x = x + width / 2
                    angle_to_rotate = self.calculate_angle_to_center(bbox_center_x)

                    if abs(angle_to_rotate) > 0.1:
                        self.rotate_to_angle(angle_to_rotate)
                    else:
                        self.move_straight(distance - 10)  # Move towards the object, stopping 10 cm before it
                        self.advance_waypoint()

    def advance_waypoint(self):
        self.current_waypoint_idx += 1
        if self.current_waypoint_idx < len(self.waypoints):
            self.target_object = self.waypoints[self.current_waypoint_idx]
            self.get_logger().info(f"Next target object: {self.target_object}")
        else:
            self.get_logger().info("All waypoints (objects) reached.")
            self.mpi_ctrl.carStop()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = YoloWaypointNavigator()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
