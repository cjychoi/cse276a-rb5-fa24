# slam_control_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import matplotlib.pyplot as plt

class SlamControlNode(Node):
    def __init__(self):
        super().__init__('slam_control_node')
        self.movement_pub = self.create_publisher(Float32MultiArray, '/movement_command', 10)
        self.twist_pub = self.create_publisher(Twist, '/twist', 10)
        self.object_sub = self.create_subscription(
            Float32MultiArray, '/ekf_slam_state', self.update_plot, 10
        )
        self.objects_to_detect = ['tv', 'bottle', 'potted plant', 'suitcase', 'umbrella', 'teddy bear', 'backpack', 'stop sign', 'oven', 'airplane']
        self.fig, self.ax = plt.subplots()
        self.set_plot_limits()
        self.robot_positions = []
        self.detected_objects = []

    def set_plot_limits(self):
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)

    def move(self, distance):
        movement_msg = Float32MultiArray()
        movement_msg.data = [distance, 0]
        self.movement_pub.publish(movement_msg)
        
        twist_msg = Twist()
        twist_msg.linear.x = 2.0
        self.twist_pub.publish(twist_msg)

    def turn(self, angle):
        movement_msg = Float32MultiArray()
        movement_msg.data = [0, np.deg2rad(angle)]
        self.movement_pub.publish(movement_msg)
        
        twist_msg = Twist()
        twist_msg.angular.z = 2.0 if angle > 0 else -2.0
        self.twist_pub.publish(twist_msg)

    def update_plot(self, msg):
        self.ax.clear()
        self.set_plot_limits()

        state = msg.data
        robot_x, robot_y, theta = state[0], state[1], state[2]
        landmarks = state[3:]

        self.ax.plot(robot_x, robot_y, 'bo-', label="Robot Path")
        for i in range(len(self.objects_to_detect)):
            x, y = landmarks[2 * i], landmarks[2 * i + 1]
            self.ax.plot(x, y, 'o', label=self.objects_to_detect[i])

        self.ax.legend(loc='lower left')
        plt.draw()
        plt.pause(0.1)

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
