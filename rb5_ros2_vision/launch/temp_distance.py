import cv2
import numpy as np
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.parameter import Parameter

class YoloCameraNode(Node):
    def __init__(self):
        super().__init__('yolo_camera_node')

        # Get parameters from the launch file
        self.declare_parameter('camera_id', '0')
        self.declare_parameter('topic_name', '/camera_0')
        self.declare_parameter('frame_rate', 30)

        # Retrieve parameters
        self.camera_id = self.get_parameter('camera_id').value
        self.topic_name = self.get_parameter('topic_name').value
        self.frame_rate = self.get_parameter('frame_rate').value

        self.bridge = CvBridge()
        self.frame = None  # To store the incoming frame

        # Target object
        self.target_object = "cell phone"

        # Load the YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # YOLOv8 nano model

        # Focal length and known width
        self.KNOWN_WIDTH = 8.0  # Width of the object in cm (adjust based on the object)
        self.FOCAL_LENGTH = 902.8  # Focal length of the camera in pixels (example)

        # Subscribe to the camera topic
        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.image_callback,
            10
        )
        # Timer to process at the desired frame rate
        self.timer = self.create_timer(1 / self.frame_rate, self.process_frame)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting ROS Image to OpenCV: {e}")

    def estimate_distance(self, bbox_width):
        if bbox_width > 0:
            distance = (self.KNOWN_WIDTH * self.FOCAL_LENGTH) / bbox_width
            return distance
        else:
            return None
    
       def process_frame(self):
        if self.frame is None:
            self.get_logger().info("No frame received yet...")
            return

        # Detect objects using YOLOv8
        results = self.model(self.frame)

        for result in results:
            detection_count = result.boxes.shape[0]
            self.get_logger().info(f'Detection count: {detection_count}')
            for i in range(detection_count):
                cls = int(result.boxes.cls[i].item())
                name = result.names[cls]
                self.get_logger().info(f'name: {name}')
                if name == self.target_object:
                    self.get_logger().info(f'Object found: {self.target_object}')

                    bounding_box = result.boxes.xyxy[i].cpu().numpy()
                    x = int(bounding_box[0])
                    y = int(bounding_box[1])
                    width = int(bounding_box[2] - x)
                    height = int(bounding_box[3] - y)

                    # Estimate the distance to the object
                    distance = self.estimate_distance(width)

                    self.temp(distance)

    def temp(self, distance):
        # Move forward distance - 10
        self.get_logger().info(f"Estimated distance to object: {distance} cm")
        self.get_logger().info("Shutting down node...")
        rclpy.shutdown()
        # Estimate the distance to the object
        distance = self.estimate_distance(width)

        self.temp(distance)

    def temp(self, distance):
        # Move forward distance - 10
        self.get_logger().info(f"Estimated distance to object: {distance} cm")
        self.get_logger().info("Shutting down node...")
        rclpy.shutdown()

    def temp2(self, distance):
        #Move forward distance - 10
        self.get_logger.info(f"Estimated distance to object: {distance} cm")
        self.get_logger().info("Shutting down node...")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    yolo_camera_node = YoloCameraNode()
    rclpy.spin(yolo_camera_node)

    # Cleanup
    rclpy.shutdown()

if __name__ == '__main__':
    main()