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
            # Process the frame using YOLO
            results = self.model(self.frame)

            for result in results:
                boxes = result.boxes.xyxy  # get the bounding boxes
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box  # unpack the box
                    object_name = self.model.names[int(cls)]
                    if object_name == self.current_object:
                        # Calculate distance to the object
                        distance = self.calculate_distance(self.KNOWN_WIDTH, (x2 - x1))  # Width of the bounding box
                        angle = self.calculate_angle(x1)  # Calculate angle to the object
                        self.get_logger().info(f"Detected {object_name} at distance {distance:.2f} cm, angle {angle:.2f} degrees.")

                        # Publish the rotation angle and movement distance
                        self.rotation_pub.publish(Float32(data=math.radians(angle)))  # Publish angle in radians
                        self.distance_pub.publish(Float32(data=distance))  # Publish distance in cm
                        break  # Stop after processing the first detected object

            # Check if the current object was found
            if self.current_object and self.current_waypoint_idx >= len(self.waypoints):
                self.get_logger().info("All objects processed. Stopping navigation.")
                self.destroy_node()  # Stop the node after processing all waypoints

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
