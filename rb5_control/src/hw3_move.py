# hw3_slam.py - Adjusted to calculate object positions relative to the robot's current position and update EKF SLAM
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist
import numpy as np
import matplotlib.pyplot as plt
import time

class MovementCommands(Node):
    def __init__(self):
        super().__init__('movement_commands_node')
        
        self.spin_and_track_pub = self.create_publisher(Float32MultiArray, '/start_spin_and_track', 10)
        
        self.update_and_plot_pub = self.create_publisher(Bool, '/start_update_and_plot', 10)
        
        self.plot_final_landmarks_pub = self.create_publisher(Bool, '/start_plot_final_landmarks', 10)

        # Subscription to flag when step is done
        self.done_sub = self.create_subscription(
            Bool, '/done_flag', self.next_step, 10
        )

        self.steps = [
            # # square
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['spin', 90],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['spin', 90],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['spin', 90],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['spin', 90],
            # # # move to start ocotogon
            ['move', 0.5],
            # # octogon
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['spin', 45],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['spin', 45],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['spin', 45],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['spin', 45],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['spin', 45],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['spin', 45],
            ['move', 0.5],
            ['move', 0.5],
            ['move', 0.5],
            ['spin', 45],
            ['move', 0.5],
            ['move', 0.5],
            ['spin', 45],
        ]
        self.step_counter = -1

    def next_step(self, msg):
        self.step_counter += 1
        print("\nnext_step")
        print(len(self.steps))
        if self.step_counter == len(self.steps):
            print("\n IF")
            self.update_and_plot()
            print("\n update and plot")
        # elif self.step_counter+1 == len(self.steps):
        #     self.plot_final_landmarks()
        elif self.step_counter < len(self.steps):
            self.spin_and_track(self.steps[self.step_counter][0], self.steps[self.step_counter][1])

    def spin_and_track(self, type, distance):
        if type == 'move':
            msg = Float32MultiArray()
            msg.data = [1.0, float(distance)]
            self.spin_and_track_pub.publish(msg)
        else:
            msg = Float32MultiArray()
            msg.data = [2.0, float(distance)]
            self.spin_and_track_pub.publish(msg)

    def update_and_plot(self):
        msg = Bool()
        msg.data = True
        self.update_and_plot_pub.publish(msg)

    def plot_final_landmarks(self):
        msg = Bool()
        msg.data = True
        self.plot_final_landmarks_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MovementCommands()
  
    print("SLAM 1")

    node.spin_and_track('move', 0.0)
    time.sleep(1)

    rclpy.spin(node)

    # # TRY 1
    # # Square movement
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
        



    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()