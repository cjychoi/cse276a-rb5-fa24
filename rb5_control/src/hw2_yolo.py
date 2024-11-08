#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
import cv2
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import math

class YoloCameraNode(Node):
    def __init__(self):
        super().__init__('yolo_camera_node')
        self.declare_parameter('camera_id', '0')
        self.declare_parameter('topic_name', '/camera_0')
        self.declare_parameter('frame_rate', 30)
        self.camera_id = self.get_parameter('camera_id').value
        self.topic_name = self.get_parameter('topic_name').value
        self.br = CvBridge()

        # Object list to detect in the specified order
        self.objects_to_detect = ['teddy bear', 'bottle']
        self.current_object_index = 0
        self.detection_timeout = 10  # Timeout after 10 seconds of searching
        self.detection_start_time = time.time()

        # YOLOv8 Model
        self.model = YOLO('yolov8n.pt')
        
        # Initialize camera feed subscription
        self.subscription = self.create_subscription(
            Image, self.topic_name, self.image_callback, 10
        )
        
        # Set up a publisher to publish distances
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)

    def image_callback(self, msg):
        """Callback function for processing the camera feed."""
        print("\n<<image_callback>>\n")
        cv_image = self.br.imgmsg_to_cv2(msg, 'bgr8')
        
        # Use YOLO model to detect objects 
        results = self.model(cv_image)
        detected_objects = results[0].boxes

        # Find and track the desired object in sequence
        for box in detected_objects:
            cls = int(box.cls.item())  # Get the class index
            object_name = self.model.names[cls]  # Map class index to object name
            if object_name == self.objects_to_detect[self.current_object_index]:
                print("\n<<Object Found!>>\n")
                self.handle_detected_object(cv_image, box)
                return

        # Timeout mechanism: if object is not found within a given time, rotate to search again
        if time.time() - self.detection_start_time > self.detection_timeout:
            print("\n<<Object Not Found - rotate_to_search>>\n")
            self.rotate_to_search()


    def handle_detected_object(self, cv_image, box):
        """Handle the detected object, calculate distance, and publish twist commands."""
        print("\n<<handle_detected_object>>\n")

        # Extract bounding box center
        x_min, y_min, x_max, y_max = box.xyxy[0]  # Access the bounding box coordinates
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min

        # Estimate distance based on object size or known dimensions
        focal_length = 902.8  # Approximate focal length for a camera
        known_width = 0.2  # Assume all objects have a known width of 20 cm
        distance = (known_width * focal_length) / width
        
        # Log detection
        self.get_logger().info(f'Detected {self.objects_to_detect[self.current_object_index]} at {distance:.2f}m')

        # Move towards the object: Adjust the robot's velocity based on distance and object position
        move_twist = Twist()
        move_twist.linear.x = float(min(distance - 0.1, 0.1))  # Move forward, stop 10 cm before the object
        move_twist.angular.z = float(-(x_center - (cv_image.shape[1] / 2)) / 500)  # Adjust heading

        self.publisher_.publish(move_twist)

        # If within a threshold distance, stop and move to the next object
        if distance < 0.2:
            self.current_object_index += 1
            self.detection_start_time = time.time()  # Reset detection timer

    def rotate_to_search(self):
        """Rotate the robot to search for objects."""
        self.get_logger().info('Rotating to search for the next object...')
        rotate_twist = Twist()
        rotate_twist.angular.z = 0.1  # Rotate at a fixed speed
        self.publisher_.publish(rotate_twist)

        # Reset detection start time after rotation
        self.detection_start_time = time.time()

class PIDcontroller(Node):
    def __init__(self, Kp, Ki, Kd):
        super().__init__("PID_Controller_NodePub")
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = None
        self.I = np.array([0.0, 0.0, 0.0])
        self.lastError = np.array([0.0, 0.0, 0.0])
        self.timestep = 0.1
        self.maximumValue = 0.02
        self.publisher_ = self.create_publisher(Twist, "/twist", 10)

    def setTarget(self, target):
        """
        Set the target pose.
        """
        self.I = np.array([0.0, 0.0, 0.0])
        self.lastError = np.array([0.0, 0.0, 0.0])
        self.target = np.array(target)

    def getError(self, currentState, targetState):
        """
        Return the difference between two states.
        """
        result = targetState - currentState
        result[2] = (result[2] + np.pi) % (2 * np.pi) - np.pi
        return result

    def update(self, currentState):
        """
        Calculate the update value on the state based on the error between current state and target state with PID.
        """
        print("\n<<Update>>\n")
        e = self.getError(currentState, self.target)
        P = self.Kp * e
        self.I += self.Ki * e * self.timestep
        D = self.Kd * (e - self.lastError)
        result = P + self.I + D
        self.lastError = e

        resultNorm = np.linalg.norm(result)
        if resultNorm > self.maximumValue:
            result = (result / resultNorm) * self.maximumValue
            self.I = 0.0

        return result


def main(args=None):
    rclpy.init(args=args)
    
    # Initialize YOLO object detector
    yolo_node = YoloCameraNode()

    # Create PID controller
    pid = PIDcontroller(0.1, 0.005, 0.005)
    
    rclpy.spin(yolo_node)
    
    yolo_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
