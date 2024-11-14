# slam_control_node.py - Modified to publish movement commands and plot EKF SLAM state
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import time

class SlamControlNode(Node):
    def __init__(self):
        super().__init__('slam_control_node')
        self.command_publisher = self.create_publisher(Float32MultiArray, '/movement_command', 10)
        self.subscription = self.create_subscription(Float32MultiArray, '/ekf_slam_state', self.update_robot_state, 10)
        
        # Initialize state for plotting
        self.robot_positions = []  # List of robot's path positions
        self.detected_objects = []  # List of detected object positions
        
        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.set_plot_limits()

    def set_plot_limits(self):
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)

    def update_robot_state(self, msg):
        # Update the robot state from SLAM state received
        self.robot_state = msg.data[:3]  # Extract x, y, theta
        self.landmarks = msg.data[3:]    # Extract landmarks from SLAM state
        
        # Track robot positions for path plotting
        self.robot_positions.append((self.robot_state[0], self.robot_state[1]))
        
        # Update detected objects based on landmark positions
        self.detected_objects = [
            (self.landmarks[i], self.landmarks[i+1], idx) 
            for idx, i in enumerate(range(0, len(self.landmarks), 2))
        ]
        
        # Print robot state
        print(f"Updated Robot State: x = {self.robot_state[0]:.2f}, y = {self.robot_state[1]:.2f}, theta = {self.robot_state[2]:.2f}")
        
        # Plot updated robot path and detected objects
        self.update_and_plot()

    def update_and_plot(self):
        self.ax.clear()
        self.set_plot_limits()
        
        # Plot robot path
        if self.robot_positions:
            self.ax.plot(*zip(*self.robot_positions), 'bo-', label="Robot Path")
        
        # Plot detected objects with unique colors
        legend_labels = {"Robot Path": self.ax.plot([], [], 'bo-', label="Robot Path")[0]}
        for x, y, obj_index in self.detected_objects:
            color = plt.cm.get_cmap('tab10', len(self.detected_objects))(obj_index)
            label = f"Object {obj_index}" if obj_index not in legend_labels else ""
            if label:
                legend_labels[label] = self.ax.plot(x, y, 'o', color=color, label=label)[0]
            else:
                self.ax.plot(x, y, 'o', color=color)

        # Add legend to the plot
        self.ax.legend(handles=legend_labels.values(), loc='lower left')
        plt.draw()
        plt.pause(0.1)
        self.save_plot()

    def save_plot(self):
        filename = 'slam_plot.png'
        self.fig.savefig(filename)
        print(f"Plot saved as {filename}")

    def publish_movement_command(self, distance=0.0, angle=0.0):
        msg = Float32MultiArray()
        msg.data = [distance, angle]
        self.command_publisher.publish(msg)

    def spin_and_track(self):
        # Define the movement pattern: Move forward and turn in a square pattern
        for _ in range(4):
            for _ in range(4):  # Stop every 0.5 meters
                self.publish_movement_command(0.5, 0.0)
                time.sleep(1)
            self.publish_movement_command(0.0, 1.57)  # 90-degree turn
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
