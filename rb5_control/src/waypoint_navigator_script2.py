import time
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from mpi_control import MegaPiController
import matplotlib.pyplot as plt

current_position = [0.0, 0.0, 0.0]

class WaypointNavigator(Node):
    global current_position

    def __init__(self):
        super().__init__('waypoint_navigator')
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)

        self.k_v = 30
        self.k_w = 55
        self.dist_per_sec = 10 / 1
        self.rad_per_sec = math.pi / 2
        self.tolerance = 0.1
        self.path_x = []
        self.path_y = []

        # Subscriptions to receive instructions from the YOLO node
        self.angle_subscriber = self.create_subscription(Float32, 'rotate_angle', self.handle_rotation, 10)
        self.distance_subscriber = self.create_subscription(Float32, 'move_distance', self.handle_movement, 10)
        self.search_subscriber = self.create_subscription(String, 'search_status', self.handle_search, 10)

    def handle_rotation(self, msg):
        self.rotate_to_angle(msg.data)

    def handle_movement(self, msg):
        self.move_straight(msg.data)

    def handle_search(self, msg):
        if msg.data == "not_found":
            # Rotate by -π/4 if the object is not found
            self.rotate_to_angle(-math.pi / 4)

    def rotate_to_angle(self, angle_diff):
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

    def move_straight(self, distance):
        movement_time = abs(distance) / (self.dist_per_sec / 100)
        self.mpi_ctrl.carStraight(self.k_v)
        time.sleep(movement_time)
        self.mpi_ctrl.carStop()

    def plot_path(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.path_x, self.path_y, marker='o', label="Robot Path")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Robot Movement Path")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    rclpy.init(args=None)
    
    navigator = WaypointNavigator()

    rclpy.spin(navigator)

    navigator.destroy_node()
    rclpy.shutdown()
