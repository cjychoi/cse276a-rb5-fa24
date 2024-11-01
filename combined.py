import cv2
import numpy as np
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import math
from mpi_control import MegaPiController
import matplotlib.pyplot as plt
import threading  # Import threading for concurrent execution

# Initialize global array to hold current position and store path history
current_position = [0.0, 0.0, 0.0]  # (x, y, theta)
position_history = []  # To store position history for plotting

class WaypointNavigator:
    global current_position, position_history  # Declare as global to track position

    def __init__(self, waypoint_file):
        # Initialize the MegaPiController in a separate thread
        self.mpi_ctrl_thread = threading.Thread(target=self.initialize_mpi)
        self.mpi_ctrl_thread.start()

        # Load waypoints from a file
        self.waypoints = self.load_waypoints(waypoint_file)
        self.current_waypoint_idx = 0

        # Control parameters prepared from calibration
        self.k_v = 30  # Speed for straight movement
        self.k_w = 55  # Speed for rotational movement
        self.dist_per_sec = 10 / 1  # 10 cm per 1 second at speed 30 for straight movement
        self.rad_per_sec = math.pi / 2  # Pi radians per 2 seconds at speed 55 for rotational movement
        self.tolerance = 0.1  # Distance tolerance to waypoint (meters)

        # Initialize plot
        plt.ion()  # Turn on interactive mode for real-time plotting
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Robot Movement Plot')
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.grid(True)

    def initialize_mpi(self):
        # This function initializes the MPI controller
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)  # Allow some time for the connection to establish

    def load_waypoints(self, filename):  # Load object waypoints from file
        waypoints = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                name, known_width = line.strip().split(',')
                waypoints.append((name, float(known_width)))
        print('waypoints: ', waypoints)
        return waypoints

    def calculate_distance(self, x1, y1, x2, y2):  # Calculate distance of goal from current location
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_angle(self, x1, y1, x2, y2):  # Calculate angular distance of goal from current rotation
        return math.atan2(y2 - y1, x2 - x1)

    def reached_waypoint(self, x_goal, y_goal):
        x, y, _ = self.get_current_position()  # Get robot's current position
        distance = self.calculate_distance(x, y, x_goal, y_goal)
        return distance < self.tolerance

    def get_current_position(self):
        return current_position[0], current_position[1], current_position[2]

    def set_current_position(self, x, y, theta):
        current_position[0] = x
        current_position[1] = y
        current_position[2] = theta
        print('Updated position: ', current_position)

    def rotate_to_angle(self, angle_diff):
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

        # Update heading after rotation
        _, _, theta = self.get_current_position()
        new_theta = theta + angle_diff
        self.set_current_position(current_position[0], current_position[1], new_theta)
        self.update_plot()

    def move_straight(self, distance):
        movement_time = abs(distance) / (self.dist_per_sec / 100)  # Convert cm/s to m/s
        self.mpi_ctrl.carStraight(self.k_v)
        time.sleep(movement_time)
        self.mpi_ctrl.carStop()

        # Update position after movement
        x, y, theta = self.get_current_position()
        delta_x = distance * math.cos(theta)
        delta_y = distance * math.sin(theta)
        new_x = x + delta_x
        new_y = y + delta_y
        self.set_current_position(new_x, new_y, theta)
        self.update_plot()

    def update_plot(self):
        # Append the current position to the history
        position_history.append((current_position[0], current_position[1]))

        # Clear the plot and re-draw the path
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_title('Robot Movement Plot')
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        
        # Plot all points from the history
        positions = np.array(position_history)
        if len(positions) > 0:
            self.ax.plot(positions[:, 0], positions[:, 1], 'b-', marker='o', markersize=5)

        plt.pause(0.001)  # Pause to update the plot


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

        # Load the YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # YOLOv8 nano model

        # Camera parameters
        self.KNOWN_WIDTH = 8.0  # Width of the object in cm (default)
        self.FOCAL_LENGTH = 902.8  # Focal length of the camera in pixels (example)

        self.CAMERA_WIDTH = 640  # Width of the camera frame in pixels
        self.CAMERA_CENTER = self.CAMERA_WIDTH / 2  # Calculate the center of the camera's field of view

        # Instantiate WaypointNavigator for robot control
        self.navigator = WaypointNavigator(waypoint_file='waypoints.txt')

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
            distance = (self.KNOWN_WIDTH * self.FOCAL_LENGTH) / bbox_width
            return distance
        else:
            return None

    def calculate_angle_to_center(self, bbox_center_x):
        offset = bbox_center_x - self.CAMERA_CENTER
        FOV = 1.0472  # Field of view of the camera in radians
        angle_to_rotate = (offset / self.CAMERA_WIDTH) * FOV
        return angle_to_rotate

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
                for waypoint in self.navigator.waypoints:
                    target_name, known_width = waypoint
                    if name == target_name:
                        self.get_logger().info(f'Object found: {target_name}')
                        bounding_box = result.boxes.xyxy[i].cpu().numpy()
                        x = int(bounding_box[0])
                        width = int(bounding_box[2] - bounding_box[0])

                        # Estimate the distance to the object
                        distance = self.estimate_distance(width)

                        # Calculate the angle to rotate towards the object
                        bbox_center_x = int(bounding_box[0] + width / 2)
                        angle_to_center = self.calculate_angle_to_center(bbox_center_x)

                        # Rotate to face the object
                        self.navigator.rotate_to_angle(angle_to_center)

                        # Move forward towards the object
                        if distance:
                            self.navigator.move_straight(distance - 0.1)

                        self.get_logger().info(f'Distance to {target_name}: {distance:.2f} cm')
                        break

    def rotate_and_search(self, angle_deg):
        angle_rad = math.radians(angle_deg)
        self.navigator.rotate_to_angle(angle_rad)
        self.process_frame()


def main(args=None):
    rclpy.init(args=args)

    yolo_node = YoloCameraNode()
    rclpy.spin(yolo_node)

    yolo_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
