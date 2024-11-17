# hw3_slam.py - Adjusted to calculate object positions relative to the robot's current position and update EKF SLAM
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time

class SlamControlNode(Node):
    def __init__(self):
        super().__init__('slam_control_node')

        self.image_update = False
        self.EKF_update = False
        
        self.movement_pub = self.create_publisher(Float32MultiArray, '/movement_command', 10)
        self.twist_pub = self.create_publisher(Twist, '/twist', 10)
        
        # Subscription to receive EKF SLAM state
        # self.ekf_state_sub = self.create_subscription(
        #     Float32MultiArray, '/ekf_slam_state', self.update_plot, 10
        # )

        self.ekf_state_sub = self.create_subscription(
            Float32MultiArray, '/ekf_slam_state', self.get_EKF_state, 10
        )

        # # Subscription to receive EKF SLAM colors
        # self.ekf_colors_sub = self.create_subscription(
        #     StringArray, '/ekf_slam_colors', self.get_colors, 10
        # )
        
        # Subscription for detected object information
        self.object_sub = self.create_subscription(
            Float32MultiArray, '/detected_object_info', self.get_image, 10
        )

        # Subscription to start spin_and_track
        self.spin_and_track_sub = self.create_subscription(
            Float32MultiArray, '/start_spin_and_track', self.spin_and_track, 10
        )

        # Subscription to start update_and_plot
        self.update_and_plot_sub = self.create_subscription(
            Bool, '/start_update_and_plot', self.update_and_plot, 10
        )

        # Subscription to start plot_final_landmarks
        self.plot_final_landmarks_sub = self.create_subscription(
            Bool, '/start_plot_final_landmarks', self.plot_final_landmarks, 10
        )

        self.covariance_sub = self.create_subscription(
            Float32MultiArray, '/ekf_slam_covariances', self.update_covariances, 10
        )

        self.done_pub = self.create_publisher(Bool, '/done_flag', 10)

        # Publisher to send updated SLAM state
        self.object_pub = self.create_publisher(Float32MultiArray, '/ekf_update', 10)

        # Publisher to send updated SLAM state
        # self.EKF_predict_pub = self.create_publisher(Float32MultiArray, '/ekf_predict', 10)

        
        self.objects_to_detect = ['laptop', 'bottle', 'potted plant', 'suitcase', 'umbrella', 'teddy bear', 'keyboard', 'stop sign', 'bird', 'cat', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
        self.error_checking = {
            "bottle": (0, -0.2),
            "stop sign": (2.1, 0),
            "potted plant": (2.1, 1.5),
            "teddy bear": (1.4, -0.3),
            "umbrella": (-0.7, 0.2),
            "keyboard": (0.4, 1.3),
            "suitcase": (-0.2, 1.9),
            "laptop": (1.3, 1.6)
        }
        # self.ekf_slam = EKFSLAM(self.objects_to_detect)
        self.fig, self.ax = plt.subplots()
        self.landmark_covariances = {}
        self.set_plot_limits()
        self.robot_positions = []  # Store estimated robot positions from EKF
        self.detected_objects = []  # Store positions of detected objects
        self.colors = plt.cm.get_cmap('tab10', len(self.objects_to_detect))
  #      self.spin_and_track()

        self.state = np.zeros((3 + 2 * len(self.objects_to_detect), 1))  # [x, y, theta, x1, y1, x2, y2, ...]

        # print("SLAM 1")
        # self.spin_and_track('move', 0.0)
        # time.sleep(1)
    
        # # TRY 1
        # # Square movement
        # for _ in range(1):
        #     for _ in range(4):  # Stop every 0.5 meters
        #         print("SLAM loop")
        #         self.spin_and_track('move', 0.5)
        #         time.sleep(1)
        #     self.spin_and_track('spin', 90)
        #     time.sleep(1)
    
        # self.update_and_plot()

    def get_colors(self, msg):
        self.colors = msg.data

    def get_EKF_state(self, msg):
        # print('\n\n\n\n\n\n\n\n++++++++++WAITED*************\n\n\n\n\n\n\n')
        self.state = msg.data
        round(self.state[0], 1)
        round(self.state[1], 1)
        # print('\n\n\n state: ', self.state, '\n\n\n')
        # print('\n\n\n msg: ', msg.data, '\n\n\n')
        
        self.EKF_update = True

        # if (self.image_update == True) and (self.EKF_update == True):
        #     self.object_callback()

    def update_covariances(self, msg):
        """ Callback to update landmark covariance data. """
        num_landmarks = len(msg.data) // 4  # Each covariance matrix is 2x2
        for i in range(num_landmarks):
            covariance_matrix = np.array(msg.data[i * 4:(i + 1) * 4]).reshape(2, 2)
            self.landmark_covariances[f"landmark_{i}"] = covariance_matrix

    def get_image(self, msg):
        # print('\n\n\n\n\n\n\n\n++++++++++IMAGE*************\n\n\n\n\n\n\n')

        self.image = msg.data
        self.image_update = True

        # if (self.image_update == True) and (self.EKF_update == True):
        #     self.object_callback()
        
    def set_plot_limits(self):
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)

    def object_callback(self):
        distance, angle, obj_index = self.image
        robot_x, robot_y, theta = self.state[0], self.state[1], self.state[2]
        obj_x = robot_x + distance * np.cos(theta + angle)
        obj_y = robot_y + distance * np.sin(theta + angle)

        # Update EKF with the world-frame coordinates of the detected object
        # self.ekf_slam.update((obj_x, obj_y), int(obj_index))
        state_msg = Float32MultiArray()
        state_msg.data = [obj_x, obj_y, obj_index]
        self.object_pub.publish(state_msg)

        object_name = self.objects_to_detect[int(obj_index)]
        print(f"\nRobot Position: (x={robot_x:.2f}, y={robot_y:.2f}, theta={theta:.2f})")
        print(f"Detected {object_name} (index {obj_index}) at world position (x={obj_x:.2f}, y={obj_y:.2f})")

        self.robot_positions.append([robot_x, robot_y])
        self.detected_objects.append((obj_x, obj_y, object_name))
        # self.update_and_plot()
        self.image_update = False
        self.EKF_update = False

    def update_plot(self):
        state_data = self.state
        robot_x, robot_y, theta = state_data[:3]
        # robot_x, robot_y, theta = robot_x[0], robot_y[0], theta[0]
        print('update plot state data', self.state)
        self.robot_positions.append([robot_x, robot_y])
        # self.detected_objects.append((obj_x, obj_y, object_name))
        
        for i in range(3, len(state_data), 2):
            obj_x, obj_y = state_data[i], state_data[i+1]
            self.detected_objects.append((obj_x, obj_y, self.objects_to_detect[(i-2)//2]))
        # self.update_and_plot()
        print('robot positions: ', self.robot_positions)
        print(' detected objects: ', self.detected_objects)

    def update_and_plot(self, msg):
        print("\nupdate and plot")
        print(self.robot_positions[1:])
        print("\n")
        print(self.detected_objects[-8:])
        print("\n")
        print(self.objects_to_detect)
        self.ax.clear()
        self.set_plot_limits()
        self.ax.plot(*zip(*self.robot_positions[1:]), 'bo-', label="Robot Path")
        
        legend_labels = {"Robot Path": self.ax.plot([], [], 'bo-', label="Robot Path")[0]}
        for x, y, name in self.detected_objects[-8:]:
            if x != 0.0 and y != 0.0:
                color = self.colors(self.objects_to_detect.index(name))
                if name not in legend_labels:
                    legend_labels[name] = self.ax.plot(x, y, 'o', color=color, label=name)[0]
                else:
                    self.ax.plot(x, y, 'o', color=color)

                
        for i in range(len(self.detected_objects[-8:])):
            # Plot covariance ellipse if available
            if f"landmark_{i}" in self.landmark_covariances:
                covariance_matrix = self.landmark_covariances[f"landmark_{i}"]
                eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

                angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                width, height = 2 * np.sqrt(eigenvalues)  # 2-sigma ellipse
                ellipse = Ellipse((x, y), width, height, angle=angle, edgecolor='blue', facecolor='none')
                self.ax.add_patch(ellipse)

        self.ax.legend(handles=legend_labels.values(), loc='lower left')
        plt.draw()
        plt.pause(0.1)
        print(self.landmark_covariances)
        self.save_plot()

        # Dictionary to store errors for each landmark
        errors = {landmark: [] for landmark in self.error_checking}

        # Calculate errors for detected landmarks
        for obj in self.detected_objects:
            # print(obj)
            # name = obj[2]  # Name of the landmark
            # detected_position = [obj[0], obj[1]]  # Detected (x, y) position
            (detected_x, detected_y, name) = obj
            # print(detected_x, detected_y, name, '\n')
            if name in self.objects_to_detect:
                true_position = self.error_checking[name]
                # print('true_position: ', true_position)
                # Compute Euclidean distance error
                error = [(detected_x - true_position[0]),(detected_y - true_position[1])]
                errors[name].append(error)
            print('errors: ', errors)
        
        # Compute average errors for each landmark
        average_errors = {name: (np.mean(vals) if vals else None) for name, vals in errors.items()}
        print(average_errors)
        # Print out average errors
        print("Average errors for landmarks:")
        for name, avg_err in average_errors.items():
            if avg_err is not None:
                print(f"{name}: {avg_err:.3f} meters")
            else:
                print(f"{name}: No detections")

        msg = Bool()
        msg.data = True
        self.done_pub.publish(msg)

    def save_plot(self):
        filename = 'slam_plot.png'
        self.fig.savefig(filename)
        print(f"Plot saved as {filename}")

    def spin_and_track(self, msg):
        type, length = msg.data
        if (type == 1.0):
            print('moving')
            self.move_forward(length)
            print('moved')
        elif (type == 2.0):
            print('spinning')
            if (length == 90):
                self.turn_90_degrees()
            elif (length == 45):
                self.turn_45_degrees()            # MAKE NEW FUNCTION FOR 45 DEGREE TURN FOR OCTOGON
            print('spun')

        # self.save_plot()
        time.sleep(1)

        # self.plot_final_landmarks()
        # self.print_final_coordinates()
        msg = Bool()
        msg.data = True
        self.done_pub.publish(msg)

    def move_forward(self, distance):
        print("Moving forward by {distance} meters")
        control_input = [distance, 0.0]
        print(control_input)
        
        # self.ekf_slam.predict(control_input)
        state_msg = Float32MultiArray()
        state_msg.data = control_input
        self.movement_pub.publish(state_msg)
        print('publisher done')
        
        move_twist = Twist()
        move_twist.linear.x = 2.0
        self.twist_pub.publish(move_twist)
        time.sleep(distance / 0.5)
        move_twist.linear.x = 0.0
        self.twist_pub.publish(move_twist)

        print('update plot')
        self.update_plot()

        # print("\nmove forward self state:")
        # print(self.state[0][0])
        # print(self.state[1][0])
        # print("\n")
        # robot_x, robot_y = self.state[0][0], self.state[1][0]
        # self.robot_positions.append([robot_x, robot_y])
        # print(f"Updated Position: x = {robot_x}, y = {robot_y}")
        # print("robot positions list: ", self.robot_positions)

    def turn_90_degrees(self):
        print("Turning 90 degrees")
        control_input = [0.0, np.pi / 2]
        # self.ekf_slam.predict(control_input)
        state_msg = Float32MultiArray()
        state_msg.data = control_input
        self.movement_pub.publish(state_msg)

        turn_twist = Twist()
        turn_twist.angular.z = 8.6
        self.twist_pub.publish(turn_twist)
        time.sleep(np.pi / 2)
        turn_twist.angular.z = 0.0
        self.twist_pub.publish(turn_twist)
        print(f"Updated Heading (theta): {self.state[2]} radians")

        print('update plot')
        self.update_plot()

    def turn_45_degrees(self):
        print("Turning 45 degrees")
        control_input = [0.0, np.pi / 4]
        # self.ekf_slam.predict(control_input)
        state_msg = Float32MultiArray()
        state_msg.data = control_input
        self.movement_pub.publish(state_msg)

        turn_twist = Twist()
        turn_twist.angular.z = 8.7
        self.twist_pub.publish(turn_twist)
        time.sleep(np.pi / 4)
        turn_twist.angular.z = 0.0
        self.twist_pub.publish(turn_twist)
        print(f"Updated Heading (theta): {self.state[2]} radians")

        print('update plot')
        self.update_plot()

    def plot_final_landmarks(self, msg):
        self.ax.clear()
        self.set_plot_limits()
        self.ax.plot(*zip(*self.robot_positions), 'bo-', label="Robot Path")

        legend_labels = {"Robot Path": self.ax.plot([], [], 'bo-', label="Robot Path")[0]}
        for x, y, name in self.detected_objects:
            if int(x) != 0 and int(y) != 0:
                color = self.colors(self.objects_to_detect.index(name))
                legend_labels[name] = self.ax.plot(x, y, 'o', color=color, label=name)[0]

        self.ax.legend(handles=legend_labels.values(), loc='lower left')
        plt.title("Robot Path and Final Detected Object Positions")
        plt.xlabel("X position (meters)")
        plt.ylabel("Y position (meters)")
        plt.show()
        self.save_plot()
        msg = Bool()
        msg.data = True
        self.done_pub.publish(msg)

    def print_final_coordinates(self):
        # print("\nFinal Coordinates of Detected Objects:")
        # for i, obj_name in enumerate(self.objects_to_detect):
        #     landmark_idx = 3 + 2 * i
        #     obj_x, obj_y = self.state[landmark_idx][0], self.state[landmark_idx + 1][0]
        #     print(f"{obj_name}: (x = {obj_x}, y = {obj_y})")

        print("\n Self State:")
        print(self.state[0], self.state[1], self.state[2])

def main(args=None):
    rclpy.init(args=args)
    node = SlamControlNode()
    print('ready')

    rclpy.spin(node)
    # print("SLAM 1")

    # node.spin_and_track('move', 0.0)
    # time.sleep(1)

    # # TRY 1
    # # # Square movement
    # for _ in range(1):
    #     for _ in range(4):  # Stop every 0.5 meters
    #         print("SLAM loop")
    #         node.spin_and_track('move', 0.5)
    #         time.sleep(1)
    #     node.spin_and_track('spin', 90)
    #     time.sleep(1)

    # node.update_and_plot()
    # node.plot_final_landmarks()

    # node.spin_and_track('move', 0.5)
    # time.sleep(1)

    # Octogon movement
    # for _ in range(8):
    #     for _ in range(2):  # Stop every 0.5 meters
    #         print("SLAM loop")
    #         node.spin_and_track('move', 0.5)
    #         time.sleep(1)
    #     node.spin_and_track('spin', 45)
    #     time.sleep(1)
        
    # try:
    #     rclpy.spin(node)
    # except KeyboardInterrupt:
    #     pass


    # # TRY 2
    # try:
    #     rclpy.spin(node)
    # except KeyboardInterrupt:
    #     pass

    # for _ in range(4):
    #     for _ in range(4):  # Stop every 0.5 meters
    #         node.spin_and_track('move', 0.5)
    #         time.sleep(1)
    #     node.spin_and_track('spin', 90)
    #     time.sleep(1)


    # # TRY 3
    # for _ in range(4):
    #     for _ in range(4):  # Stop every 0.5 meters
    #         node.spin_and_track('move', 0.5)
    #         time.sleep(1)
    #         # rclpy.spin_once(node)
    #         # time.sleep(1)
    #     node.spin_and_track('spin', 90)
    #     time.sleep(1)
    #     # rclpy.spin_once(node)
    #     # time.sleep(1)


    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
