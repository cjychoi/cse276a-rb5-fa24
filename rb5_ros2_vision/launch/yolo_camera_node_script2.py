import cv2
import numpy as np
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
import math

class YoloCameraNode(Node):
    def __init__(self):
        super().__init__('yolo_camera_node')

        # Initialize parameters
        self.declare_parameter('camera_id', '0')
        self.declare_parameter('topic_name', '/camera_0')
        self.declare_parameter('frame_rate', 30)
        self.declare_parameter('object_file', 'object.txt')

        # Retrieve parameters
        self.camera_id = self.get_parameter('camera_id').value
        self.topic_name = self.get_parameter('topic_name').value
        self.frame_rate = self.get_parameter('frame_rate').value
        self.object_file = self.get_parameter('object_file').value

        # Initialize variables
        self.bridge = CvBridge()
        self.frame = None
        self.current_object = None
        self.KNOWN_WIDTH = None
        self.is_moving = False  # Track if the robot is moving
        self.load_waypoints()

        # Load the YOLO model
        self.model = YOLO('yolov8n.pt')

        # Camera parameters
        self.FOCAL_LENGTH = 902.8
        self.CAMERA_WIDTH = 640
        self.CAMERA_CENTER = self.CAMERA_WIDTH / 2

        # Subscribe to the camera topic
        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.image_callback,
            10
        )

        # Subscribe to the is_moving flag
        self.is_moving_sub = self.create_subscription(
            Bool,
            '/is_moving',
            self.update_is_moving,
            10
        )

        # Timer for frame processing
        self.timer = self.create_timer(1 / self.frame_rate, self.process_frame)

        # Publisher for sending rotation and distance to the waypoint navigator
        self.rotation_pub = self.create_publisher(Float32, '/rotation_angle', 10)
        self.distance_pub = self.create_publisher(Float32, '/move_distance', 10)

    def load_waypoints(self):
        # Load the waypoints from the object file
        with open(self.object_file, 'r') as f:
            self.waypoints = [line.strip().split(', ') for line in f]
        self.current_waypoint_idx = 0
        self.load_next_waypoint()

    def load_next_waypoint(self):
        # Load the next object from the object file
        if self.current_waypoint_idx < len(self.waypoints):
            self.current_object, self.KNOWN_WIDTH, _ = self.waypoints[self.current_waypoint_idx]
            self.KNOWN_WIDTH = float(self.KNOWN_WIDTH)
            self.current_waypoint_idx += 1
            self.get_logger().info(f"Looking for object: {self.current_object}")

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def update_is_moving(self, msg):
        # Update the is_moving flag based on the received message
        self.is_moving = msg.data

    def process_frame(self):
        if self.frame is not None and not self.is_moving:
            if self.frame is None:
                self.get_logger().info("No frame received yet...")
                return
            if self.is_moving:
                self.get_logger().info("Currently moving. Waiting for movement to complete...")
                return  # Skip processing if currently moving
                
            # Process the frame using YOLO
            results = self.model(self.frame)

            object_found = False  # Flag to track if the object was found
            for result in results:
                for i in range(result.boxes.shape[0]):
                    cls = int(result.boxes.cls[i].item())
                    name = result.names[cls]
                    if name == self.current_object:
                        # Object detected, calculate distance and angle to rotate
                        bbox = result.boxes.xyxy[i].cpu().numpy()
                        x, width = int(bbox[0]), int(bbox[2] - bbox[0])
                        distance = self.estimate_distance(width)
                        bbox_center_x = x + width / 2
                        angle_to_rotate = self.calculate_angle_to_center(bbox_center_x)
                        if abs(angle_to_rotate) > 0.1:
                            # Rotate towards the object
                            self.rotate_to_angle(angle_to_rotate)
                        else:
                            # Move forward if facing the object
                            self.get_logger().info(f"Moving forward by {distance - 10} cm.")
                            self.move_forward(distance - 10)
                            # Here, you would typically set up a mechanism to call `movement_complete()` after the move is done
                        self.load_next_waypoint()  # Load the next waypoint after processing this one
                        object_found = True  # Set the flag to true since the object was found
                        break

            if not object_found:
                # If the object was not found, rotate -pi/4 radians
                self.get_logger().info(f"Object '{self.current_object}' not found. Rotating -pi/4 radians.")
                self.rotate_to_angle(-math.pi / 4)  # Rotate by -pi/4 radians if the object is not found

    def calculate_distance(self, known_width, pixel_width):
        return (self.FOCAL_LENGTH * known_width) / pixel_width

    def calculate_angle(self, x1):
        return (self.CAMERA_CENTER - x1) / self.CAMERA_CENTER * 90  # Assuming 90 degrees field of view

def main(args=None):
    rclpy.init(args=args)
    yolo_camera_node = YoloCameraNode()

    try:
        rclpy.spin(yolo_camera_node)
    except KeyboardInterrupt:
        yolo_camera_node.get_logger().info("Shutting down...")

    yolo_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
