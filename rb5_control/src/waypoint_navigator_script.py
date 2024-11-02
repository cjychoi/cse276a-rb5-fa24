import time
import math
import rclpy
from rclpy.node import Node
from mpi_control import MegaPiController
import matplotlib.pyplot as plt

current_position = [0.0, 0.0, 0.0]

class WaypointNavigator(Node):
    global current_position

    def __init__(self, waypoint_file):
        super().__init__('navigator')
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
                x, y, theta = map(float, line.strip().split(','))
                waypoints.append((x, y, theta))
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

    def navigate_to_waypoint(self, x_goal, y_goal, theta_goal):
        while not self.reached_waypoint(x_goal, y_goal):
            x, y, theta = self.get_current_position()
            distance = self.calculate_distance(x, y, x_goal, y_goal)
            angle_to_goal = self.calculate_angle(x, y, x_goal, y_goal)
            angle_diff = angle_to_goal - theta

            if abs(angle_diff) > 0.1:
                self.rotate_to_angle(angle_diff)
                self.set_current_position([x, y, angle_to_goal])
            else:
                self.move_straight(distance)
                angle_to_goal = self.calculate_angle(x, y, x_goal, y_goal)
                angle_diff = theta_goal - angle_to_goal
                self.rotate_to_angle(angle_diff)
                self.set_current_position([x_goal, y_goal, theta_goal])

    def plot_path(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.path_x, self.path_y, marker='o', label="Robot Path")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Robot Movement Path")
        plt.legend()
        plt.grid(True)
        plt.show()

    def start_navigation(self):
        for waypoint in self.waypoints:
            x_goal, y_goal, theta_goal = waypoint
            if waypoint != current_position:
                self.navigate_to_waypoint(x_goal, y_goal, theta_goal)
                self.set_current_position(waypoint)
        print("All waypoints reached.")
        self.plot_path()

if __name__ == "__main__":
    rclpy.init(args=None)
    
    navigator = WaypointNavigator(waypoint_file='object.txt')

    navigator.start_navigation()

    rclpy.spin(navigator)

    navigator.destroy_node()
    rclpy.shutdown()
