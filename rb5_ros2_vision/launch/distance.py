import cv2
import numpy as np
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.parameter import Parameter
# from control import WaypointNavigator  # Import the navigation class
import math
from mpi_control import MegaPiController

current_position = [0.0, 0.0, 0.0]        # initialize global array to hold current position

class YoloCameraNode(Node):
    def __init__(self):
        # Movements
        # Initialize the MegaPiController
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)  # Allow some time for the connection to establish

        # Load waypoints from a file
        self.waypoints = self.load_waypoints('waypoints.txt')
        self.current_waypoint_idx = 0

        # Control parameters prepared from calibration
        self.k_v = 30  # Speed for straight movement
        self.k_w = 55  # Speed for rotational movement
        self.dist_per_sec = 10 / 1  # 10 cm per 1 second at speed 30 for straight movement   
        self.rad_per_sec = math.pi / 2  # Pi radians per 2 seconds at speed 55 for rotational movement
        self.tolerance = 0.1  # Distance tolerance to waypoint (meters)

        def load_waypoints(self, filename):        # load waypoints from file
            waypoints = []
            with open(filename, 'r') as f:        # open file, read waypoints line-by-line, put into array of arrays
                for line in f.readlines():
                    x, y, theta = map(float, line.strip().split(','))      
                    waypoints.append((x, y, theta))
            print('waypoints: ', waypoints)
            return waypoints

        
        # Object Detection
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
        self.target_object = "person"

        # Load the YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # YOLOv8 nano model

        # Focal length and known width
        self.KNOWN_WIDTH = 8.0  # Width of the object in cm (adjust based on the object)
        self.FOCAL_LENGTH = 902.8  # Focal length of the camera in pixels (example)

        # Camera parameters for calculating the center
        self.CAMERA_WIDTH = 640  # Width of the camera frame in pixels
        self.CAMERA_CENTER = self.CAMERA_WIDTH / 2  # Calculate the center of the camera's field of view

#        # Instantiate WaypointNavigator for robot control
#        self.navigator = WaypointNavigator(waypoint_file='waypoints.txt')

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

    def calculate_angle_to_center(self, bbox_center_x):
        # Calculate the offset of the bounding box center from the camera's center
        offset = bbox_center_x - self.CAMERA_CENTER
        
        # Field of view of the camera in radians (example: 60 degrees -> 1.0472 radians)
        FOV = 1.0472
        
        # Calculate the angle to rotate based on the offset and field of view
        angle_to_rotate = (offset / self.CAMERA_WIDTH) * FOV
        return angle_to_rotate

    def process_frame(self):
        if self.frame is None:
            self.get_logger().info("No frame received yet...")
            return

        # Detect objects using YOLOv8
        results = self.model(self.frame)
        
        # Check if any objects were detected
        if len(results) == 0 or results[0].boxes.shape[0] == 0:
            self.get_logger().info("Object not found. Rotating -45 degrees.")
            self.rotate_and_search(-45)
            return

        # If there are detections, process them
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

                    # Calculate the bounding box center
                    bbox_center_x = x + width / 2
                    
                    # Calculate the angle to rotate to center the object
                    angle_to_rotate = self.calculate_angle_to_center(bbox_center_x)

                    if abs(angle_to_rotate) > 0.1:  # If the object is not centered
                        self.temp2(angle_to_rotate)
                        self.rotate_to_angel(angle_to_rotate)
                    else:
                        self.get_logger().info(f"Estimated distance to object: {distance} cm")
                        self.move_straight(distance - 10)

    def rotate_and_search(self, degrees):
        # Convert degrees to radians
        radians_to_rotate = math.radians(degrees)
        self.get_logger().info(f"Rotating by {degrees} degrees ({radians_to_rotate} radians).")
        
        # Rotate the robot
        self.rotate_to_angle(radians_to_rotate)
        
        # After rotating, try to search for the object again
        self.get_logger().info("Searching for object again after rotation.")

    def rotate_to_angle(self, angle_diff):    # rotate robot to rotational goal by amount of time
        # Calculate rotation time based on angle difference
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

    def move_straight(self, distance):        # move robot straight by amount of time
        # Calculate movement time based on distance
        movement_time = abs(distance) / (self.dist_per_sec / 100)  # Convert cm/s to m/s
        self.mpi_ctrl.carStraight(self.k_v)
        time.sleep(movement_time)
        self.mpi_ctrl.carStop()
        

    def temp2(self, angle_to_rotate):
        # Rotate the robot based on the angle
        self.get_logger().info(f"Rotating robot by {angle_to_rotate} radians to center object.")
#        self.navigator.rotate_to_angle(angle_to_rotate)

def main(args=None):
    rclpy.init(args=args)
    yolo_camera_node = YoloCameraNode()
    rclpy.spin(yolo_camera_node)

    # Cleanup
    yolo_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
