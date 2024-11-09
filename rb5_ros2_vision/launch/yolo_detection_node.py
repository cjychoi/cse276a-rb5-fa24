# yolo_detection_node.py
import torch
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np

class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')
        self.declare_parameter('camera_id', '0')
        self.declare_parameter('topic_name', '/camera_0')
        self.camera_id = self.get_parameter('camera_id').value
        self.topic_name = self.get_parameter('topic_name').value
        self.br = CvBridge()

        # Objects to detect and their known widths (in meters)
        self.objects_to_detect = {
            'teddy bear': 0.2,
            'backpack': 0.3,
            'umbrella': 0.6,
            'bottle': 0.1
        }

        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')

        # Subscription for camera feed
        self.subscription = self.create_subscription(
            Image, self.topic_name, self.image_callback, 10
        )

        # Publisher for object detection results
        self.publisher_ = self.create_publisher(Float32MultiArray, '/detected_object_info', 10)

    def image_callback(self, msg):
        cv_image = self.br.imgmsg_to_cv2(msg, 'bgr8')
        
        # Perform YOLO object detection on the image
        results = self.model(cv_image)
        detected_objects = results[0].boxes

        for box in detected_objects:
            cls = int(box.cls.item())
            object_name = self.model.names[cls]
            if object_name in self.objects_to_detect:
                self.get_logger().info(f'\n<<{object_name} Detected!>>\n')
                self.handle_detected_object(cv_image, box, object_name)

    def handle_detected_object(self, cv_image, box, object_name):
        # Extract object details
        x_min, y_min, x_max, y_max = box.xyxy[0]
        x_center = (x_min + x_max) / 2
        width = x_max - x_min

        # Calculate distance to object
        focal_length = 902.8  # Example focal length value (in pixels)
        known_width = self.objects_to_detect[object_name]
        distance = (known_width * focal_length) / width

        # Calculate angle based on current rotation and image center offset
        angle_offset = -(x_center - (cv_image.shape[1] / 2)) / 500

        # Publish object info (distance and angle)
        msg = Float32MultiArray()
        msg.data = [distance, angle_offset]
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    yolo_node = YoloDetectionNode()

    try:
        rclpy.spin(yolo_node)
    except KeyboardInterrupt:
        pass

    yolo_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()