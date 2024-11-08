import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
import cv2
from ultralytics import YOLO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting

# EKF SLAM Class
class EKFSLAM:
    def __init__(self, objects_to_detect, focal_length, known_width):
        self.state = np.zeros((3 + 2 * len(objects_to_detect), 1))  # [x, y, theta, x1, y1, x2, y2, ...]
        self.P = np.eye(3 + 2 * len(objects_to_detect)) * 1e-3  # Initial covariance
        self.objects_to_detect = objects_to_detect
        self.focal_length = focal_length
        self.known_width = known_width

    def predict(self, u):
        theta = self.state[2, 0]
        G = np.eye(len(self.state))  
        G[0, 2] = -u[0] * np.sin(theta)
        G[1, 2] = u[0] * np.cos(theta)
        Q = np.eye(len(self.state)) * 1e-3  
        self.state[:3] += u  
        self.P = G @ self.P @ G.T + Q

    def update(self, z, landmark_index):
        H = np.zeros((2, len(self.state))) 
        H[0, 3 + 2 * landmark_index] = 1
        H[1, 3 + 2 * landmark_index + 1] = 1
        R = np.eye(2) * 1e-3  

        z = z.reshape(2, 1)
        state_slice = self.state[3 + 2 * landmark_index: 3 + 2 * (landmark_index + 1)]
        innovation = z - state_slice

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state += (K @ innovation).reshape(self.state.shape)
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P

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

        self.objects_to_detect = {
            'teddy bear': 0.2,
            'backpack': 0.3,
            'umbrella': 0.6,
            'bottle': 0.1
        }
        self.detected_objects = set()  # Track detected objects
        self.detection_timeout = 10
        self.starting_angle = None

        self.model = YOLO('yolov8n.pt')

        self.subscription = self.create_subscription(
            Image, self.topic_name, self.image_callback, 10
        )
        
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)

        # Initialize EKF
        self.ekf_slam = EKFSLAM(
            list(self.objects_to_detect.keys()),
            focal_length=902.8,
            known_width=0.2
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

        # Start spinning and detection
        self.spin_and_track()

    def image_callback(self, msg):
        cv_image = self.br.imgmsg_to_cv2(msg, 'bgr8')
        
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
        # Move the robot in a 2m x 2m square with YOLO detection happening while it moves
        for i in range(4):  # For a square
            self.move_forward(2.0)  # Move 2.0 meters with continuous detection
            self.turn_90_degrees()  # Turn 90 degrees with continuous detection

        # After completing the square, stop the robot and ensure final plot
        self.save_plot()

    def move_forward(self, distance):
        # Move the robot forward by a specified distance (in meters)
        move_twist = Twist()
        move_twist.linear.x = 2.0  # Set a forward speed (1.0 m/s)
        self.publisher_.publish(move_twist)

        # Continuously move forward while updating position and checking for object detection
        start_time = time.time()
        while time.time() - start_time < distance / 1.0:
            # YOLO detection happens during movement
            self.run_yolo_detection()

            # Update the robot's position and plot it
            self.plot_robot_positions()

            # Allow short sleep for smoother operation
            time.sleep(0.1)

        move_twist.linear.x = 0.0  # Stop the robot
        self.publisher_.publish(move_twist)

    def turn_90_degrees(self):
        # Rotate the robot 90 degrees (assuming constant speed)
        turn_twist = Twist()
        turn_twist.angular.z = 8.5  # Set a rotation speed (1.0 rad/s)
        self.publisher_.publish(turn_twist)

        # Continuously turn while checking for object detection
        start_time = time.time()
        while time.time() - start_time < 1.57:  # 1.57 seconds for 90 degrees
            # YOLO detection happens during the turn
            self.run_yolo_detection()

            # Update the robot's position and plot it
            self.plot_robot_positions()

            # Allow short sleep for smoother operation
            time.sleep(0.1)

        turn_twist.angular.z = 0.0  # Stop rotating
        self.publisher_.publish(turn_twist)

    def run_yolo_detection(self):
        # YOLO detection method is called continuously during movement or rotation
        # This method checks the camera input and detects objects using YOLO
        self.get_logger().info('Running YOLO detection...')
        
        # Add code to capture the camera frame and detect objects
        # The image_callback will be automatically called when Image messages are received

        # Detected objects are updated in self.handle_detected_object

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
            if obj_name in self.detected_objects:
                x, y = state[3 + 2 * i], state[3 + 2 * i + 1]
                self.object_lines[i].set_data(x, y)
        
        # Save the updated plot to file for later analysis
        self.fig.savefig('robot_movement.png')

    def save_plot(self):
        # Save the final plot of robot movement and detected objects
        self.fig.savefig('final_robot_movement.png')
        self.get_logger().info('Plot saved as final_robot_movement.png')

def main(args=None):
    rclpy.init(args=args)
    node = YoloCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped cleanly')
    except BaseException:
        node.get_logger().error('Exception in node:', exc_info=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
