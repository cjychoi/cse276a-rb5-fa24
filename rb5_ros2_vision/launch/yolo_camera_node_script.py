import cv2
import numpy as np
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math
from mpi_control import MegaPiController

class YoloCameraNode(Node):
    def __init__(self):
        super().__init__('yolo_camera_node')

        # Parameters from launch file
        self.declare_parameter('camera_id', '0')
        self.declare_parameter('topic_name', '/camera_0')
        self.declare_parameter('frame_rate', 30)

        self.camera_id = self.get_parameter('camera_id').value
        self.topic_name = self.get_parameter('topic_name').value
        self.frame_rate = self.get_parameter('frame_rate').value

        self.bridge = CvBridge()
        self.frame = None

        # Target object
        self.target_object = "cell phone"

        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')

        # Known object dimensions and camera parameters
        self.KNOWN_WIDTH = 8.0
        self.FOCAL_LENGTH = 902.8
        self.CAMERA_WIDTH = 640
        self.CAMERA_CENTER = self.CAMERA_WIDTH / 2

        # ROS subscription to camera topic
        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.image_callback,
            10
        )

        # Frame processing timer
        self.timer = self.create_timer(1 / self.frame_rate, self.process_frame)

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting ROS Image to OpenCV: {e}")

    def estimate_distance(self, bbox_width):
        if bbox_width > 0:
            return (self.KNOWN_WIDTH * self.FOCAL_LENGTH) / bbox_width
        return None

    def calculate_angle_to_center(self, bbox_center_x):
        offset = bbox_center_x - self.CAMERA_CENTER
        FOV = 1.0472  # Example 60 degrees in radians
        return (offset / self.CAMERA_WIDTH) * FOV

    def rotate_to_angle(self, radians_to_rotate):
        self.get_logger().info(f"Rotating by {radians_to_rotate} radians.")
        navigator.rotate_to_angle(radians_to_rotate)

    def process_frame(self):
        if self.frame is None:
            self.get_logger().info("No frame received yet...")
            return

        results = self.model(self.frame)
        if len(results) == 0 or results[0].boxes.shape[0] == 0:
            self.get_logger().info("Object not found. Rotating -45 degrees.")
            self.rotate_to_angle(math.radians(-45))
            return

        for result in results:
            for i in range(result.boxes.shape[0]):
                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]
                if name == self.target_object:
                    bbox = result.boxes.xyxy[i].cpu().numpy()
                    x, width = int(bbox[0]), int(bbox[2] - bbox[0])
                    distance = self.estimate_distance(width)
                    bbox_center_x = x + width / 2
                    angle_to_rotate = self.calculate_angle_to_center(bbox_center_x)

                    if abs(angle_to_rotate) > 0.1:
                        self.rotate_to_angle(angle_to_rotate)
                    else:
                        self.get_logger().info(f"Moving forward by {distance - 10} cm.")
                        navigator.move_straight(distance - 10)
                    break

def main(args=None):
    rclpy.init(args=args)
    yolo_camera_node = YoloCameraNode()
    rclpy.spin(yolo_camera_node)

    yolo_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
