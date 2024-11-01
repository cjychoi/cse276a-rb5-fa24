import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from mpi_control import MegaPiController

# Initialize a global variable to store the robot's path
robot_path = []

class YoloCameraNode(Node):
    def __init__(self):
        super().__init__('yolo_camera_node')
        self.declare_parameter('camera_id', '0')
        self.declare_parameter('topic_name', '/camera_0')
        self.declare_parameter('frame_rate', 30)

        self.camera_id = self.get_parameter('camera_id').value
        self.topic_name = self.get_parameter('topic_name').value
        self.frame_rate = self.get_parameter('frame_rate').value

        self.bridge = CvBridge()
        self.frame = None
        self.target_object = "person"
        self.model = YOLO('yolov8n.pt')

        self.KNOWN_WIDTH = 8.0
        self.FOCAL_LENGTH = 902.8
        self.CAMERA_WIDTH = 640
        self.CAMERA_CENTER = self.CAMERA_WIDTH / 2

        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.image_callback,
            10
        )

        self.timer = self.create_timer(1 / self.frame_rate, self.process_frame)

    def image_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting ROS Image to OpenCV: {e}")

    def estimate_distance(self, bbox_width):
        return (self.KNOWN_WIDTH * self.FOCAL_LENGTH) / bbox_width if bbox_width > 0 else None

    def calculate_angle_to_center(self, bbox_center_x):
        offset = bbox_center_x - self.CAMERA_CENTER
        FOV = 1.0472
        return (offset / self.CAMERA_WIDTH) * FOV

    def process_frame(self):
        if self.frame is None:
            return

        results = self.model(self.frame)
        if len(results) == 0 or results[0].boxes.shape[0] == 0:
            self.rotate_and_search(-45)
            return

        for result in results:
            detection_count = result.boxes.shape[0]
            for i in range(detection_count):
                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]
                if name == self.target_object:
                    bounding_box = result.boxes.xyxy[i].cpu().numpy()
                    x, y = int(bounding_box[0]), int(bounding_box[1])
                    width = int(bounding_box[2] - x)

                    distance = self.estimate_distance(width)
                    bbox_center_x = x + width / 2
                    angle_to_rotate = self.calculate_angle_to_center(bbox_center_x)
                    return distance, angle_to_rotate

    def rotate_and_search(self, degrees):
        radians_to_rotate = math.radians(degrees)
        self.get_logger().info(f"Rotating by {degrees} degrees ({radians_to_rotate} radians).")
        self.temp2(radians_to_rotate)

    def temp2(self, angle_to_rotate):
        self.get_logger().info(f"Rotating robot by {angle_to_rotate} radians to center object.")


class WaypointNavigator:
    def __init__(self, waypoint_file, yolo_node):
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)
        self.waypoints = self.load_waypoints(waypoint_file)
        self.current_waypoint_idx = 0

        self.k_v = 30
        self.k_w = 55
        self.dist_per_sec = 10 / 1
        self.rad_per_sec = math.pi / 2
        self.tolerance = 0.1
        self.yolo_node = yolo_node

    def load_waypoints(self, filename):
        waypoints = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                x, y, theta = map(float, line.strip().split(','))
                waypoints.append((x, y, theta))
        return waypoints

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

    def navigate_to_goal(self):
        distance, angle_to_rotate = self.yolo_node.process_frame()
        if distance and angle_to_rotate:
            if abs(angle_to_rotate) > 0.1:
                self.rotate_to_angle(angle_to_rotate)
            else:
                self.move_straight(distance - 10)
                robot_path.append((distance, angle_to_rotate))  # Track position

    def plot_movement(self):
        x, y = zip(*[(pt[0], pt[1]) for pt in robot_path])
        plt.plot(x, y, marker='o')
        plt.title("Robot Movement Path")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    yolo_camera_node = YoloCameraNode()
    navigator = WaypointNavigator('waypoints.txt', yolo_camera_node)
    while navigator.current_waypoint_idx < len(navigator.waypoints):
        navigator.navigate_to_goal()
    navigator.plot_movement()
    yolo_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
