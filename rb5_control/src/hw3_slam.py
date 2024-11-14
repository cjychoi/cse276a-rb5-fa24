# slam_control_node.py - Control Node for Robot and SLAM Interaction
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
        self.publisher_detection = self.create_publisher(Float32MultiArray, '/control/detection_info', 10)
        self.publisher_movement = self.create_publisher(Float32MultiArray, '/control/movement_info', 10)
        self.subscription_robot_state = self.create_subscription(Float32MultiArray, '/slam/robot_state', self.robot_state_callback, 10)
        self.subscription_object_state = self.create_subscription(Float32MultiArray, '/slam/object_states', self.object_state_callback, 10)
        
        self.fig, self.ax = plt.subplots()
        self.set_plot_limits()
        self.robot_positions = []
        self.detected_objects = []

    def set_plot_limits(self):
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)

    def robot_state_callback(self, msg):
        x, y, theta = msg.data
        self.robot_positions.append([x, y])
        self.update_and_plot()

    def object_state_callback(self, msg):
        object_states = msg.data
        self.detected_objects = [(object_states[2 * i], object_states[2 * i + 1], obj) for i, obj in enumerate(self.objects_to_detect)]
        self.update_and_plot()

    def send_detection_info(self, obj_x, obj_y, obj_index):
        detection_msg = Float32MultiArray()
        detection_msg.data = [obj_x, obj_y, float(obj_index)]
        self.publisher_detection.publish(detection_msg)

    def send_movement_info(self, distance, angle):
        movement_msg = Float32MultiArray()
        movement_msg.data = [distance, angle]
        self.publisher_movement.publish(movement_msg)

    def update_and_plot(self):
        self.ax.clear()
        self.set_plot_limits()
        self.ax.plot(*zip(*self.robot_positions), 'bo-', label="Robot Path")
        legend_labels = {"Robot Path": self.ax.plot([], [], 'bo-', label="Robot Path")[0]}
        
        for x, y, name in self.detected_objects:
            color = 'orange'  # Simplified color for each unique object
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
