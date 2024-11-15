# hw3_slam_control.py - Adjusted to calculate object positions relative to the robot's current position and update EKF SLAM
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import matplotlib.pyplot as plt
import time

class SlamControlNode(Node):
    def __init__(self):
        super().__init__('slam_control_node')

        self.image_update = False
        self.EKF_update = False
        self.state = None  # Initialize self.state as None
        
        self.movement_pub = self.create_publisher(Float32MultiArray, '/movement_command', 10)
        self.twist_pub = self.create_publisher(Twist, '/twist', 10)
        
        # Subscription to receive EKF SLAM state
        self.ekf_state_sub = self.create_subscription(
            Float32MultiArray, '/ekf_slam_state', self.get_EKF_state, 10
        )
        
        # Subscription for detected object information
        self.object_sub = self.create_subscription(
            Float32MultiArray, '/detected_object_info', self.get_image, 10
        )

        # Publisher to send updated SLAM state
        self.EKF_update_pub = self.create_publisher(Float32MultiArray, '/ekf_update', 10)

        # Publisher to send predicted SLAM state
        self.EKF_predict_pub = self.create_publisher(Float32MultiArray, '/ekf_predict', 10)

        self.objects_to_detect = ['tv', 'bottle', 'potted plant', 'suitcase', 'umbrella', 'teddy bear', 'backpack', 'stop sign', 'oven', 'airplane']
        self.fig, self.ax = plt.subplots()
        self.set_plot_limits()
        self.robot_positions = []  # Store estimated robot positions from EKF
        self.detected_objects = []  # Store positions of detected objects, with optional names
        self.colors = plt.cm.get_cmap('tab10', len(self.objects_to_detect))

    def get_EKF_state(self, msg):
        # Set self.state with the data from the EKF SLAM node
        self.state = np.array(msg.data)  # Store as a NumPy array for easy indexing
        self.EKF_update = True
        self.update_plot()

        if self.image_update and self.EKF_update:
            self.object_callback()

    def get_image(self, msg):
        self.image = msg.data
        self.image_update = True

        if self.image_update and self.EKF_update:
            self.object_callback()

    def set_plot_limits(self):
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)

    def object_callback(self):
        if self.state is None:
            return  # Ensure that self.state is set before proceeding

        distance, angle, obj_index = self.image
        robot_x, robot_y, theta = self.state[0], self.state[1], self.state[2]
        obj_x = robot_x + distance * np.cos(theta + angle)
        obj_y = robot_y + distance * np.sin(theta + angle)

        state_msg = Float32MultiArray()
        state_msg.data = [float(obj_x), float(obj_y), float(obj_index)]
        self.EKF_update_pub.publish(state_msg)

        object_name = self.objects_to_detect[int(obj_index)]
        print(f"Robot Position: (x={robot_x:.2f}, y={robot_y:.2f}, theta={theta:.2f})")
        print(f"Detected {object_name} at world position (x={obj_x:.2f}, y={obj_y:.2f})")

        self.robot_positions.append([robot_x, robot_y])
        self.detected_objects.append((obj_x, obj_y, object_name))
        self.update_and_plot()

    def update_plot(self):
        if self.state is None:
            return  # Ensure self.state is set before plotting

        robot_x, robot_y, theta = self.state[:3]
        self.robot_positions.append([robot_x, robot_y])
        
        for i in range(3, len(self.state), 2):
            obj_x, obj_y = self.state[i], self.state[i+1]
            self.detected_objects.append((obj_x, obj_y, 'Unknown'))  # Default name for objects without labels
        self.update_and_plot()

    def update_and_plot(self):
        self.ax.clear()
        self.set_plot_limits()
        self.ax.plot(*zip(*self.robot_positions), 'bo-', label="Robot Path")

        legend_labels = {"Robot Path": self.ax.plot([], [], 'bo-', label="Robot Path")[0]}
        for x, y, name in self.detected_objects:
            color = self.colors(self.objects_to_detect.index(name)) if name in self.objects_to_detect else 'gray'
            if name not in legend_labels:
                legend_labels[name] = self.ax.plot(x, y, 'o', color=color, label=name)[0]
            else:
                self.ax.plot(x, y, 'o', color=color)

        self.ax.legend(handles=legend_labels.values(), loc='lower left')
        plt.draw()
        plt.pause(0.1)
        self.save_plot()

    def save_plot(self):
        filename = 'slam_plot.png'
        self.fig.savefig(filename)
        print(f"Plot saved as {filename}")

    def spin_and_track(self, type, length):
        if type == 'move':
            print('Moving forward')
            self.move_forward(length)
            print('Moved forward')
        elif type == 'spin':
            print('Spinning')
            if length == 90:
                self.turn_90_degrees()
            elif length == 45:
                self.turn_45_degrees()
            print('Spun')

        self.save_plot()
        time.sleep(1)

        self.plot_final_landmarks()
        self.print_final_coordinates()

    def move_forward(self, distance):
        print("Moving forward by 0.5 meters")
        control_input = [distance, 0.0]
        
        # Send movement command to EKF SLAM node
        state_msg = Float32MultiArray()
        state_msg.data = control_input
        self.EKF_predict_pub.publish(state_msg)

        move_twist = Twist()
        move_twist.linear.x = 2.0
        self.twist_pub.publish(move_twist)
        time.sleep(distance / 0.5)
        move_twist.linear.x = 0.0
        self.twist_pub.publish(move_twist)

        if self.state is not None:
            robot_x, robot_y = self.state[0], self.state[1]
            self.robot_positions.append([robot_x, robot_y])
            print(f"Updated Position: x = {robot_x}, y = {robot_y}")

    def turn_90_degrees(self):
        print("Turning 90 degrees")
        control_input = [0.0, np.pi / 2]
        
        # Send movement command to EKF SLAM node
        state_msg = Float32MultiArray()
        state_msg.data = control_input
        self.EKF_predict_pub.publish(state_msg)

        turn_twist = Twist()
        turn_twist.angular.z = 8.0
        self.twist_pub.publish(turn_twist)
        time.sleep(np.pi / 2)
        turn_twist.angular.z = 0.0
        self.twist_pub.publish(turn_twist)

        if self.state is not None:
            print(f"Updated Heading (theta): {self.state[2]} radians")

    def plot_final_landmarks(self):
        self.ax.clear()
        self.set_plot_limits()
        self.ax.plot(*zip(*self.robot_positions), 'bo-', label="Robot Path")

        legend_labels = {"Robot Path": self.ax.plot([], [], 'bo-', label="Robot Path")[0]}
        for x, y, name in self.detected_objects:
            color = self.colors(self.objects_to_detect.index(name)) if name in self.objects_to_detect else 'gray'
            legend_labels[name] = self.ax.plot(x, y, 'o', color=color, label=name)[0]

        self.ax.legend(handles=legend_labels.values(), loc='lower left')
        plt.title("Robot Path and Final Detected Object Positions")
        plt.xlabel("X position (meters)")
        plt.ylabel("Y position (meters)")
        plt.show()
        self.save_plot()

    def print_final_coordinates(self):
        print("\nFinal Coordinates of Detected Objects:")
        for i, obj_name in enumerate(self.objects_to_detect):
            landmark_idx = 3 + 2 * i
            if self.state is not None:
                obj_x, obj_y = self.state[landmark_idx], self.state[landmark_idx + 1]
                print(f"{obj_name}: (x = {obj_x:.2f}, y = {obj_y:.2f})")

def main(args=None):
    rclpy.init(args=args)
    node = SlamControlNode()
    print("SLAM 1")


    node.spin_and_track('move', 0.0)
    time.sleep(2)
    

    # Movement pattern example
    # for _ in range(4):
    #     for _ in range(4):  # Stop every 0.5 meters
    #         node.spin_and_track('move', 0.5)
    #         time.sleep(1)
    #     node.spin_and_track('spin', 90)
    #     time.sleep(1)
        
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()