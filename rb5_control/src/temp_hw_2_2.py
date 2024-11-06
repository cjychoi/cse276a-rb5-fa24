import cv2
import numpy as np
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math
import time
from mpi_control import MegaPiController

# YoloCameraNode class to detect objects and calculate distance and angle
class YoloCameraNode(Node):
    def __init__(self):
        super().__init__('yolo_camera_node')

        # Parameters
        self.declare_parameter('camera_id', '0')
        self.declare_parameter('topic_name', '/camera_0')
        self.declare_parameter('frame_rate', 30)
        self.camera_id = self.get_parameter('camera_id').value
        self.topic_name = self.get_parameter('topic_name').value
        self.frame_rate = self.get_parameter('frame_rate').value
        
        self.bridge = CvBridge()
        self.frame = None

        # Target object will be dynamically set per waypoint
        self.target_object = None

        # Load the YOLOv8 model
        self.model = YOLO('yolov8n.pt')

        # Focal length and known width
        self.KNOWN_WIDTH = 8.0
        self.FOCAL_LENGTH = 902.8

        # Camera parameters
        self.CAMERA_WIDTH = 640
        self.CAMERA_CENTER = self.CAMERA_WIDTH / 2

        # Robot navigation
        self.navigator = WaypointNavigator(waypoint_file='waypoints.txt')

        # Camera subscription
        self.subscription = self.create_subscription(Image, self.topic_name, self.image_callback, 10)

        # Timer for frame processing
        self.timer = self.create_timer(1 / self.frame_rate, self.process_frame)

    def image_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting ROS Image to OpenCV: {e}")

    def estimate_distance(self, bbox_width):
        if bbox_width > 0:
            return (self.KNOWN_WIDTH * self.FOCAL_LENGTH) / bbox_width
        return None

    def calculate_angle_to_center(self, bbox_center_x):
        offset = bbox_center_x - self.CAMERA_CENTER
        FOV = 1.0472  # Field of view of the camera in radians
        return (offset / self.CAMERA_WIDTH) * FOV

    def process_frame(self):
        if self.frame is None:
            self.get_logger().info("No frame received yet...")
            return

        # Detect objects using YOLOv8
        results = self.model(self.frame)

        # Check for target object detection
        for result in results:
            detection_count = result.boxes.shape[0]
            for i in range(detection_count):
                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]
                if name == self.target_object:
                    self.get_logger().info(f'Object found: {self.target_object}')
                    bounding_box = result.boxes.xyxy[i].cpu().numpy()
                    x = int(bounding_box[0])
                    width = int(bounding_box[2] - x)

                    # Estimate distance and calculate the angle to rotate
                    distance = self.estimate_distance(width)
                    bbox_center_x = x + width / 2
                    angle_to_rotate = self.calculate_angle_to_center(bbox_center_x)

                    if abs(angle_to_rotate) > 0.1:
                        self.temp2(angle_to_rotate)
                    else:
                        self.temp(distance)

    def temp(self, distance):
        self.get_logger().info(f"Moving forward, distance to object: {distance - 10} cm")
        self.navigator.move_straight(distance - 10)

    def temp2(self, angle_to_rotate):
        self.get_logger().info(f"Rotating {angle_to_rotate} radians to center the object.")
        self.navigator.rotate_to_angle(angle_to_rotate)

# WaypointNavigator class to handle robot control
class WaypointNavigator:
    def __init__(self, waypoint_file):
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)
        self.waypoints = self.load_waypoints(waypoint_file)
        self.current_waypoint_idx = 0
        self.k_v = 30  # Speed for straight movement
        self.k_w = 55  # Speed for rotational movement
        self.dist_per_sec = 10 / 1
        self.rad_per_sec = math.pi / 2

    def load_waypoints(self, filename):
        waypoints = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                waypoints.append(line.strip())
        return waypoints

    def move_straight(self, distance):
        movement_time = abs(distance) / (self.dist_per_sec / 100)
        self.mpi_ctrl.carStraight(self.k_v)
        time.sleep(movement_time)
        self.mpi_ctrl.carStop()

    def rotate_to_angle(self, angle_diff):
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

    def get_waypoints(self):
        return self.waypoints

# Main function to control the process
def main(args=None):
    rclpy.init(args=args)
    yolo_camera_node = YoloCameraNode()
    
    # Loop through the waypoints and process each object
    waypoints = yolo_camera_node.navigator.get_waypoints()
    
    for waypoint in waypoints:
        yolo_camera_node.get_logger().info(f"Processing waypoint: {waypoint}")
        
        # Set the target object
        yolo_camera_node.target_object = waypoint

        # Find object, rotate to angle, and move forward
        yolo_camera_node.get_logger().info(f"Searching for {waypoint}...")
        
        # Assume process_frame will handle the detection, rotation, and movement

    # Cleanup
    yolo_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
