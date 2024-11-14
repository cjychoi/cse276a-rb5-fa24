# slam_control_node.py - Adjusted to communicate with EKFSLAM via publishers and subscribers
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
        # Publisher for movement commands to EKFSLAM node
        self.movement_command_pub = self.create_publisher(Float32MultiArray, '/movement_command', 10)
        
        # Subscriber for detected objects
        self.subscription = self.create_subscription(
            Float32MultiArray, '/detected_object_info', self.object_callback, 10
        )

        # Subscriber for SLAM state updates from EKFSLAM node
        self.slam_state_sub = self.create_subscription(
            Float32MultiArray, '/ekf_slam_state', self.slam_state_callback, 10
        )

        self.objects_to_detect = ['tv', 'bottle', 'potted plant', 'suitcase', 'umbrella', 'teddy bear', 'backpack', 'stop sign', 'oven', 'airplane']
        self.fig, self.ax = plt.subplots()
        self.set_plot_limits()
        self.robot_positions = []  # Store estimated robot positions from SLAM
        self.detected_objects = []  # Store positions of detected objects

    def set_plot_limits(self):
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)

    def slam_state_callback(self, msg):
        # SLAM state updates (received from EKFSLAM node)
        robot_x, robot_y, theta = msg.data[:3]
        landmarks = msg.data[3:]

        # Update robot position and detected objects for plotting
        self.robot_positions.append([robot_x, robot_y])
        self.detected_objects = [
            (landmarks[i], landmarks[i + 1], idx)
            for idx, i in enumerate(range(0, len(landmarks), 2))
        ]

        print(f"Updated Robot Position from SLAM: x={robot_x:.2f}, y={robot_y:.2f}, theta={theta:.2f}")
        self.update_and_plot()

    def object_callback(self, msg):
        # Handles new object detections and sends coordinates to EKFSLAM node
        distance, angle, obj_index = msg.data
        robot_x, robot_y, theta = self.robot_positions[-1] if self.robot_positions else (0, 0, 0)
        obj_x = robot_x + distance * np.cos(theta + angle)
        obj_y = robot_y + distance * np.sin(theta + angle)

        # Publish the object’s world-frame coordinates to EKFSLAM node
        slam_update_msg = Float32MultiArray()
        slam_update_msg.data = [obj_x, obj_y, obj_index]
        self.movement_command_pub.publish(slam_update_msg)
        
        object_name = self.objects_to_detect[int(obj_index)]
        print(f"Detected {object_name} at world position (x={obj_x:.2f}, y={obj_y:.2f})")

    def update_and_plot(self):
        self.ax.clear()
        self.set_plot_limits()
        self.ax.plot(*zip(*self.robot_positions), 'bo-', label="Robot Path")

        legend_labels = {"Robot Path": self.ax.plot([], [], 'bo-', label="Robot Path")[0]}
        for x, y, name in self.detected_objects:
            color = plt.cm.get_cmap('tab10', len(self.objects_to_detect))(name)
            if name not in legend_labels:
                legend_labels[name] = self.ax.plot(x, y, 'o', color=color, label=self.objects_to_detect[name])[0]
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

    def publish_movement_command(self, distance=0.0, angle=0.0):
        # Send movement command to EKFSLAM node
        msg = Float32MultiArray()
        msg.data = [distance, angle]
        self.movement_command_pub.publish(msg)

    def spin_and_track(self):
        # Square movement pattern
        for _ in range(4):
            for _ in range(4):
                self.publish_movement_command(0.5, 0.0)  # Move forward 0.5 meters
                time.sleep(1)
            self.publish_movement_command(0.0, np.pi / 2)  # 90-degree turn
            time.sleep(1)
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = SlamControlNode()
    try:
        node.spin_and_track()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
