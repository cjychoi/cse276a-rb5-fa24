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
        self.movement_pub = self.create_publisher(Float32MultiArray, '/movement_command', 10)
        self.twist_pub = self.create_publisher(Twist, '/twist', 10)
        
        # Subscription to receive EKF SLAM state
        self.ekf_state_sub = self.create_subscription(
            Float32MultiArray, '/ekf_slam_state', self.update_plot, 10
        )
        
        # Subscription for detected object information
        self.object_sub = self.create_subscription(
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
        if type == 'move':
            self.move_forward(length)
        elif type == 'spin' and length == 90:
            self.turn_90_degrees()

    def move_forward(self, distance):
        movement_msg = Float32MultiArray()
        movement_msg.data = [distance, 0]
        self.movement_pub.publish(movement_msg)

    def turn_90_degrees(self):
        movement_msg = Float32MultiArray()
        movement_msg.data = [0, np.pi / 2]
        self.movement_pub.publish(movement_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SlamControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()