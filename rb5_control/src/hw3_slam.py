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
<<<<<<< HEAD
        self.movement_pub = self.create_publisher(Float32MultiArray, '/movement_command', 10)
        self.twist_pub = self.create_publisher(Twist, '/twist', 10)
        
        # Subscription to receive EKF SLAM state
        self.ekf_state_sub = self.create_subscription(
            Float32MultiArray, '/ekf_slam_state', self.update_plot, 10
        )
        
        # Subscription for detected object information
        self.object_sub = self.create_subscription(
=======
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)
        self.subscription = self.create_subscription(
>>>>>>> f008a836d8a9e5d7b538d3ee15197b07624dd090
            Float32MultiArray, '/detected_object_info', self.object_callback, 10
        )
        
        self.objects_to_detect = ['tv', 'bottle', 'potted plant', 'suitcase', 'umbrella', 'teddy bear', 'backpack', 'stop sign', 'oven', 'airplane']
        self.fig, self.ax = plt.subplots()
        self.set_plot_limits()
        self.robot_positions = []  # Store estimated robot positions from EKF
        self.detected_objects = []  # Store positions of detected objects

    def set_plot_limits(self):
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)

    def object_callback(self, msg):
        # Here we assume distance and angle are relative to the robot's current position
        distance, angle, obj_index = msg.data
        movement_msg = Float32MultiArray()
        movement_msg.data = [distance, angle, obj_index]
        self.movement_pub.publish(movement_msg)  # Publish movement command to EKF node

    def update_plot(self, msg):
        state_data = msg.data
        robot_x, robot_y, theta = state_data[:3]
        self.robot_positions.append([robot_x, robot_y])
        
        for i in range(3, len(state_data), 2):
            obj_x, obj_y = state_data[i], state_data[i+1]
            self.detected_objects.append((obj_x, obj_y))

        self.update_and_plot()

    def update_and_plot(self):
        self.ax.clear()
        self.set_plot_limits()
        self.ax.plot(*zip(*self.robot_positions), 'bo-', label="Robot Path")

        # Plot each detected object at its calculated position
        for obj in self.detected_objects:
            obj_x, obj_y = obj
            self.ax.plot(obj_x, obj_y, 'ro')

        self.ax.legend(loc='lower left')
        plt.draw()
        plt.pause(0.1)
        self.save_plot()

    def save_plot(self):
        filename = 'slam_plot.png'
        self.fig.savefig(filename)
        print(f"Plot saved as {filename}")

    def spin_and_track(self, type, length):
<<<<<<< HEAD
        if type == 'move':
=======
        
        # self.move_forward(0.5)
        # self.save_plot()
        # time.sleep(1)

        # self.move_forward(0.5)
        # self.save_plot()
        # time.sleep(1)
        
        # for _ in range(4):
        #     for _ in range(4):  # Stop every 0.5 meters
        #         self.move_forward(0.5)
        #         self.save_plot()
        #         time.sleep(1)
        #     self.turn_90_degrees()
        #     self.save_plot()
        #     time.sleep(1)

        if (type == 'move'):
>>>>>>> f008a836d8a9e5d7b538d3ee15197b07624dd090
            self.move_forward(length)
        elif type == 'spin' and length == 90:
            self.turn_90_degrees()

    def move_forward(self, distance):
<<<<<<< HEAD
        movement_msg = Float32MultiArray()
        movement_msg.data = [distance, 0]
        self.movement_pub.publish(movement_msg)

    def turn_90_degrees(self):
        movement_msg = Float32MultiArray()
        movement_msg.data = [0, np.pi / 2]
        self.movement_pub.publish(movement_msg)
=======
        print("Moving forward by 0.5 meters")
        control_input = [distance, 0]
        self.ekf_slam.predict(control_input)

        move_twist = Twist()
        move_twist.linear.x = 2.0
        self.publisher_.publish(move_twist)
        time.sleep(distance / 0.5)
        move_twist.linear.x = 0.0
        self.publisher_.publish(move_twist)

        robot_x, robot_y = self.ekf_slam.state[0, 0], self.ekf_slam.state[1, 0]
        self.robot_positions.append([robot_x, robot_y])
        print(f"Updated Position: x = {robot_x}, y = {robot_y}")

    def turn_90_degrees(self):
        print("Turning 90 degrees")
        control_input = [0, np.pi / 2]
        self.ekf_slam.predict(control_input)

        turn_twist = Twist()
        turn_twist.angular.z = 8.0
        self.publisher_.publish(turn_twist)
        time.sleep(np.pi / 2)
        turn_twist.angular.z = 0.0
        self.publisher_.publish(turn_twist)

        print(f"Updated Heading (theta): {self.ekf_slam.state[2, 0]} radians")

    def plot_final_landmarks(self):
        self.ax.clear()
        self.set_plot_limits()
        self.ax.plot(*zip(*self.robot_positions), 'bo-', label="Robot Path")

        legend_labels = {"Robot Path": self.ax.plot([], [], 'bo-', label="Robot Path")[0]}
        for x, y, name in self.detected_objects:
            color = self.ekf_slam.colors(self.objects_to_detect.index(name))
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
            obj_x, obj_y = self.ekf_slam.state[landmark_idx, 0], self.ekf_slam.state[landmark_idx + 1, 0]
            print(f"{obj_name}: (x = {obj_x:.2f}, y = {obj_y:.2f})")
>>>>>>> f008a836d8a9e5d7b538d3ee15197b07624dd090

def main(args=None):
    rclpy.init(args=args)
    node = SlamControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


    print("Start")

    # TRY 1
    for _ in range(4):
        for _ in range(4):  # Stop every 0.5 meters
            node.spin_and_track('move', 0.5)
            time.sleep(1)
        node.spin_and_track('spin', 90)
        time.sleep(1)
        
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


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