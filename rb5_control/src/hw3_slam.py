# hw3_slam_control.py - Adjusted to communicate with EKFSLAMNode for SLAM updates
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
        self.publisher_ = self.create_publisher(Float32MultiArray, '/movement_command', 10)
        self.subscription = self.create_subscription(
            Float32MultiArray, '/ekf_slam_state', self.state_callback, 10
        )
        self.object_subscription = self.create_subscription(
            Float32MultiArray, '/detected_object_info', self.object_callback, 10
        )

        self.fig, self.ax = plt.subplots()
        self.robot_positions, self.detected_objects = [], []
        self.set_plot_limits()

    def set_plot_limits(self):
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)

    def state_callback(self, msg):
        state = np.array(msg.data)
        robot_x, robot_y = state[0], state[1]
        self.robot_positions.append([robot_x, robot_y])

        # Plot updated path
        self.update_and_plot()

    def object_callback(self, msg):
        distance, angle, obj_index = msg.data
        obj_index = int(obj_index)

        control_msg = Float32MultiArray()
        control_msg.data = [distance, angle, obj_index]
        self.publisher_.publish(control_msg)

    def spin_and_track(self, type, length):
        control_msg = Float32MultiArray()
        control_msg.data = [length, 0 if type == 'move' else np.pi / 2]
        self.publisher_.publish(control_msg)
        self.save_plot()
        time.sleep(1)

    def update_and_plot(self):
        self.ax.clear()
        self.set_plot_limits()
        self.ax.plot(*zip(*self.robot_positions), 'bo-', label="Robot Path")

        for x, y, name in self.detected_objects:
            color = plt.cm.tab10(self.objects_to_detect.index(name) / len(self.objects_to_detect))
            self.ax.plot(x, y, 'o', color=color, label=name)

        self.ax.legend(loc='lower left')
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
        for _ in range(4):
            for _ in range(4):
                node.spin_and_track('move', 0.5)
                time.sleep(1)
            node.spin_and_track('spin', 90)
            time.sleep(1)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
