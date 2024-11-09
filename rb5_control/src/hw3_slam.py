import torch
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt
import threading
import time

# Placeholder for EKF SLAM class
class EKFSLAM:
    def __init__(self, object_list, focal_length, known_width):
        self.known_width = known_width
        self.focal_length = focal_length
        # Initialize SLAM state (robot pose and landmarks)
        self.state = np.zeros((3 + 2 * len(object_list), 1))

    def update(self, measurement, obj_index):
        # Placeholder for EKF update
        pass

    def get_state(self):
        return self.state

# YOLO Camera Node with EKF-SLAM Integration
class YoloCameraNode(Node):
    def __init__(self):
        super().__init__('yolo_camera_node')
        self.declare_parameter('camera_id', '0')
        self.declare_parameter('topic_name', '/camera_0')
        self.declare_parameter('frame_rate', 30)
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
        self.detected_objects = set()  # Track detected objects
        self.detection_timeout = 10  # Timeout for stopping detection of already detected objects

        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')

        # Subscription for camera feed
        self.subscription = self.create_subscription(
            Image, self.topic_name, self.image_callback, 10
        )

        # Publisher for robot motion (Twist)
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)

        # Initialize EKF
        self.ekf_slam = EKFSLAM(
            list(self.objects_to_detect.keys()),
            focal_length=902.8,  # Example focal length value (in pixels)
            known_width=0.2  # Example known width (in meters)
        )

        # Initialize plot for real-time visualization
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-5, 5)  # Adjust limits based on environment size
        self.ax.set_ylim(-5, 5)
        self.robot_line, = self.ax.plot([], [], 'bo', label="Robot")
        self.object_lines = [self.ax.plot([], [], 'ro', label=obj)[0] for obj in self.objects_to_detect.keys()]
        self.robot_data = np.array([0, 0])
        self.object_data = {obj: [] for obj in self.objects_to_detect.keys()}  # Store positions of detected objects
        self.robot_positions = []  # Store the robot's position history

        # Start YOLO detection in a separate thread
        self.start_detection()

        # Start the movement after YOLO detection starts
        self.spin_and_track()

    def start_detection(self):
        # Start object detection in a separate thread
        detection_thread = threading.Thread(target=self.spin_yolo_detection)
        detection_thread.start()

    def spin_yolo_detection(self):
        # YOLO object detection loop
        rclpy.spin(self)

    def image_callback(self, msg):
        cv_image = self.br.imgmsg_to_cv2(msg, 'bgr8')
        
        # Perform YOLO object detection on the image
        results = self.model(cv_image)
        detected_objects = results[0].boxes

        for box in detected_objects:
            cls = int(box.cls.item())
            object_name = self.model.names[cls]
            if object_name in self.objects_to_detect and object_name not in self.detected_objects:
                self.get_logger().info(f'\n<<{object_name} Found!>>\n')
                self.handle_detected_object(cv_image, box, object_name)

    def handle_detected_object(self, cv_image, box, object_name):
        # Extract object details
        x_min, y_min, x_max, y_max = box.xyxy[0]
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min

        # Calculate distance to object
        distance = (self.ekf_slam.known_width * self.ekf_slam.focal_length) / width

        # Calculate angle based on current rotation and image center offset
        angle_offset = -(x_center - (cv_image.shape[1] / 2)) / 500

        # Update EKF with the detected object's relative position
        obj_index = list(self.objects_to_detect.keys()).index(object_name)
        self.ekf_slam.update(np.array([distance, angle_offset]), obj_index)

        # Mark the object as detected so it won't be processed again
        self.detected_objects.add(object_name)

    def spin_and_track(self):
        # Move the robot in a 2m x 2m square
        for i in range(4):  # For a square
            self.move_forward(2.0)  # Move 2.0 meters
            self.turn_90_degrees()  # Turn 90 degrees

        # After completing the square, stop the robot and plot its path
        self.plot_robot_positions()

        # Save the plot
        self.save_plot()

    def move_forward(self, distance):
        # Move the robot forward by a specified distance (in meters)
        move_twist = Twist()
        move_twist.linear.x = 2.0  # Set a faster forward speed (1.0 m/s)
        self.publisher_.publish(move_twist)
        time.sleep(distance / 1.0 * 4)  # Move for the time required based on speed
        move_twist.linear.x = 0.0  # Stop the robot
        self.publisher_.publish(move_twist)

    def turn_90_degrees(self):
        # Rotate the robot 90 degrees (assuming constant speed)
        turn_twist = Twist()
        turn_twist.angular.z = 8.5  # Set a faster rotation speed (1.0 rad/s)
        self.publisher_.publish(turn_twist)
        time.sleep(1.57)  # 90 degrees = 1.57 radians, so it takes 1.57 seconds at 1.0 rad/s
        turn_twist.angular.z = 0.0  # Stop rotating
        self.publisher_.publish(turn_twist)

    def plot_robot_positions(self):
        # Update the robot's position and plot it
        state = self.ekf_slam.get_state()
        robot_x, robot_y, _ = state[0, 0], state[1, 0], state[2, 0]
        self.robot_positions.append([robot_x, robot_y])  # Add the robot's new position to the list

        # Plot all robot positions together in blue
        robot_positions_array = np.array(self.robot_positions)
        self.robot_line.set_data(robot_positions_array[:, 0], robot_positions_array[:, 1])

        # Plot object positions
        for i, obj_name in enumerate(self.objects_to_detect.keys()):
            if obj_name in self.detected_objects:  # Only plot detected objects
                object_positions_array = np.array(self.object_data[obj_name])
                if object_positions_array.size > 0:  # Ensure array is non-empty
                    self.object_lines[i].set_data(object_positions_array[:, 0], object_positions_array[:, 1])

    def save_plot(self):
        self.fig.savefig('slam_plot.png')  # Save the plot as an image file

def main(args=None):
    rclpy.init(args=args)
    yolo_camera_node = YoloCameraNode()

    try:
        rclpy.spin(yolo_camera_node)
    except KeyboardInterrupt:
        pass

    yolo_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
