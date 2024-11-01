import cv2
import numpy as np
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math
from rb5_controller.pid_controller import PIDcontroller

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
        self.target_object = "cell phone"

        self.model = YOLO('yolov8n.pt')
        self.KNOWN_WIDTH = 8.0  
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

        # Timer to process at the desired frame rate
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
        FOV = 1.0472  
        return (offset / self.CAMERA_WIDTH) * FOV

    def process_frame(self):
        if self.frame is None:
            self.get_logger().info("No frame received yet...")
            return

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
                    bounding_box = result.boxes.xyxy[i].cpu().numpy()
                    x = int(bounding_box[0])
                    width = int(bounding_box[2] - x)
                    distance = self.estimate_distance(width)
                    bbox_center_x = x + width / 2
                    angle_to_rotate = self.calculate_angle_to_center(bbox_center_x)

                    if abs(angle_to_rotate) > 0.1:
                        self.publish_pid_rotation(angle_to_rotate)
                    else:
                        self.publish_pid_forward(distance - 10)

    def rotate_and_search(self, degrees):
        radians_to_rotate = math.radians(degrees)
        self.get_logger().info(f"Rotating by {degrees} degrees.")
        self.publish_pid_rotation(radians_to_rotate)

    def publish_pid_rotation(self, angle):
        pid_controller = PIDcontroller(0.02, 0.005, 0.005)
        pid_controller.setTarget([0, 0, angle])

    def publish_pid_forward(self, distance):
        pid_controller = PIDcontroller(0.02, 0.005, 0.005)
        pid_controller.setTarget([distance, 0, 0])

def main(args=None):
    rclpy.init(args=args)
    yolo_camera_node = YoloCameraNode()
    rclpy.spin(yolo_camera_node)

    yolo_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
