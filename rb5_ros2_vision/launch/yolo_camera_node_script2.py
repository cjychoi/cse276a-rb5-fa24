import time
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from mpi_control import MegaPiController
import matplotlib.pyplot as plt

current_position = [0.0, 0.0, 0.0]  # Initialize global array to hold the current position
path = []  # To store the robot's path

class WaypointNavigator(Node):
    def __init__(self, waypoint_file):
        super().__init__('waypoint_navigator')
        self.get_logger().info("Starting waypoint navigator...")

        # Initialize the MegaPiController
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)

        # Load waypoints from file
        self.waypoints = self.load_waypoints(waypoint_file)
        self.current_waypoint_idx = 0

        # Control parameters prepared from calibration
        self.k_v = 30
        self.k_w = 55
        self.dist_per_sec = 10 / 1
        self.rad_per_sec = math.pi / 2
        self.tolerance = 0.1

        # Subscriptions to angle and distance from the YOLO node
        self.angle_sub = self.create_subscription(Float32, 'rotate_angle', self.handle_rotate_angle, 10)
        self.distance_sub = self.create_subscription(Float32, 'move_distance', self.handle_move_distance, 10)
        self.rotation_request_sub = self.create_subscription(String, 'rotation_request', self.handle_rotation_request, 10)

        self.final_pose = None

        # Initialize plot
        plt.ion()

    def load_waypoints(self, filename):
        waypoints = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                x, y, theta = map(float, line.strip().split(','))
                waypoints.append((x, y, theta))
        self.get_logger().info(f"Waypoints loaded: {waypoints}")
        return waypoints

    def get_current_position(self):
        return current_position[0], current_position[1], current_position[2]

    def set_current_position(self, waypoint):
        current_position[0] = waypoint[0]
        current_position[1] = waypoint[1]
        current_position[2] = waypoint[2]
        path.append((current_position[0], current_position[1]))
        self.get_logger().info(f"Set current position: {waypoint}")
        self.plot_path()

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

    def handle_rotate_angle(self, msg):
        angle_diff = msg.data
        self.rotate_to_angle(angle_diff)

    def handle_move_distance(self, msg):
        distance = msg.data
        self.move_straight(distance)
        self.set_current_position([current_position[0] + distance, current_position[1], current_position[2]])

    def handle_rotation_request(self, msg):
        if msg.data == '-pi/4':
            self.rotate_to_angle(-math.pi / 4)

    def plot_path(self):
        plt.clf()
        x_coords, y_coords = zip(*path)
        plt.plot(x_coords, y_coords, marker='o')
        plt.title("Robot Path")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.pause(0.01)

def main(args=None):
    rclpy.init(args=args)
    navigator = WaypointNavigator(waypoint_file='waypoints.txt')
    rclpy.spin(navigator)

    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
