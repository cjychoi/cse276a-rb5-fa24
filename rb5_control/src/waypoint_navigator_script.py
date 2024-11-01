import time
import math
from mpi_control import MegaPiController
import matplotlib.pyplot as plt

current_position = [0.0, 0.0, 0.0]

class WaypointNavigator:
    global current_position

    def __init__(self, waypoint_file):
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)
        self.waypoints = self.load_waypoints(waypoint_file)
        self.current_waypoint_idx = 0
        self.k_v = 30
        self.k_w = 55
        self.dist_per_sec = 10 / 1
        self.rad_per_sec = math.pi / 2
        self.tolerance = 0.1
        self.path_x = []
        self.path_y = []

    def load_waypoints(self, filename):
        waypoints = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                object_name, size = line.strip().split(',')
                waypoints.append((object_name, float(size)))
        return waypoints

    def calculate_distance(self, x1, y1, x2, y2):
        return (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) / 2

    def calculate_angle(self, x1, y1, x2, y2):
        if (x1 == x2) and (y1 != y2):
            return 0 if (y2 > y1) else math.pi
        elif (y1 == y2) and (x1 != x2):
            return math.pi / 2 if (x1 > x2) else -(math.pi / 2)
        return math.atan2(y2 - y1, x2 - x1) - math.pi/2

    def reached_waypoint(self, x_goal, y_goal):
        x, y, _ = self.get_current_position()
        distance = self.calculate_distance(x, y, x_goal, y_goal)
        return distance < self.tolerance

    def get_current_position(self):
        return current_position[0], current_position[1], current_position[2]

    def set_current_position(self, waypoint):
        current_position[0], current_position[1], current_position[2] = waypoint
        self.path_x.append(waypoint[0])
        self.path_y.append(waypoint[1])

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

    def start_navigation(self, yolo_node):
        for object_name, known_width in self.waypoints:
            self.get_logger().info(f"Setting target object to {object_name} with known width {known_width}")
            yolo_node.load_target(object_name, known_width)
            # Perform navigation with the updated target object
            # Assume navigation logic here follows the target identified by YoloCameraNode
            # Update position tracking
            # (Add rotation and distance calculation code if needed)
            self.set_current_position(current_position)
        
        print("All waypoints processed.")
        self.plot_path()

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
    navigator = WaypointNavigator(waypoint_file='object.txt')
    yolo_camera_node = YoloCameraNode()
    navigator.start_navigation(yolo_camera_node)
