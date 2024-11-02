# imports
import time
import math
import rclpy                          # ROS2 Python client library
from rclpy.node import Node           # Node class for ROS2
from std_msgs.msg import String, Float32  # Message types
from mpi_control import MegaPiController  # Import your motor controller

current_position = [0.0, 0.0, 0.0]    # Initialize global array to hold the current position

class WaypointNavigator(Node):         # ROS2 node class
    def __init__(self, waypoint_file):
        super().__init__('waypoint_navigator')   # Initialize the ROS2 node with a name
        self.get_logger().info("Starting waypoint navigator...")

        # Initialize the MegaPiController
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)  # Allow some time for the connection to establish

        # Load waypoints from a file
        self.waypoints = self.load_waypoints(waypoint_file)
        self.current_waypoint_idx = 0

        # Control parameters prepared from calibration
        self.k_v = 30  # Speed for straight movement
        self.k_w = 55  # Speed for rotational movement
        self.dist_per_sec = 10 / 1  # 10 cm per 1 second at speed 30 for straight movement   
        self.rad_per_sec = math.pi / 2  # Pi radians per 2 seconds at speed 55 for rotational movement
        self.tolerance = 0.1  # Distance tolerance to waypoint (meters)

        # Create publishers and subscribers if needed
        self.pub = self.create_publisher(String, 'nav_status', 10)  # Publish navigation status
        self.timer = self.create_timer(0.5, self.publish_status)  # Periodically publish status

        # Start navigation
        self.start_navigation()

    def load_waypoints(self, filename):  # Load waypoints from file
        waypoints = []
        with open(filename, 'r') as f:  # Open file, read waypoints line-by-line, put into array of arrays
            for line in f.readlines():
                x, y, theta = map(float, line.strip().split(','))
                waypoints.append((x, y, theta))
        self.get_logger().info(f"Waypoints loaded: {waypoints}")
        return waypoints

    def calculate_distance(self, x1, y1, x2, y2):  # Calculate distance to goal from current position
        return (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) / 2

    def calculate_angle(self, x1, y1, x2, y2):  # Calculate angular difference from current rotation to goal
        if (x1 == x2) and (y1 != y2):  # Goal position is on x-axis
            return 0 if y2 > y1 else math.pi
        elif (y1 == y2) and (x1 != x2):  # Goal position is on y-axis
            return math.pi / 2 if x1 > x2 else -(math.pi / 2)
        else:  # Both x and y movements needed
            return math.atan2(y2 - y1, x2 - x1) - math.pi / 2

    def reached_waypoint(self, x_goal, y_goal):  # Check if the robot has reached the waypoint
        x, y, _ = self.get_current_position()  # Get the robot's current position
        distance = self.calculate_distance(x, y, x_goal, y_goal)
        return distance < self.tolerance

    def get_current_position(self):  # Get the robot's current position from global array
        return current_position[0], current_position[1], current_position[2]  # x, y, theta

    def set_current_position(self, waypoint):  # Store the robot's current position in the global array
        current_position[0] = waypoint[0]
        current_position[1] = waypoint[1]
        current_position[2] = waypoint[2]
        self.get_logger().info(f"Set current position: {waypoint}")

    def rotate_to_angle(self, angle_diff):  # Rotate robot to the goal angle
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

    def move_straight(self, distance):  # Move robot straight by the calculated distance
        movement_time = abs(distance) / (self.dist_per_sec / 100)  # Convert cm/s to m/s
        self.mpi_ctrl.carStraight(self.k_v)
        time.sleep(movement_time)
        self.mpi_ctrl.carStop()

    def navigate_to_waypoint(self, x_goal, y_goal, theta_goal):  # Navigate to x, y, theta goal
        while not self.reached_waypoint(x_goal, y_goal):  # While not in position, check goal vs. current position
            x, y, theta = self.get_current_position()

            # Calculate distance and angle to the goal
            distance = self.calculate_distance(x, y, x_goal, y_goal)
            angle_to_goal = self.calculate_angle(x, y, x_goal, y_goal)
            angle_diff = angle_to_goal - theta

            if abs(angle_diff) > 0.1:  # Rotate first if not facing the goal
                self.rotate_to_angle(angle_diff)
                self.set_current_position([x, y, angle_to_goal])
            else:  # Move straight if facing the goal
                self.move_straight(distance)
                # Rotate to final theta goal
                angle_to_goal = self.calculate_angle(x, y, x_goal, y_goal)
                angle_diff = theta_goal - angle_to_goal
                self.rotate_to_angle(angle_diff)
                self.set_current_position([x_goal, y_goal, theta_goal])

    def start_navigation(self):  # Start navigation process
        for waypoint in self.waypoints:  # Iterate through each waypoint
            x_goal, y_goal, theta_goal = waypoint
            self.get_logger().info(f"Navigating to waypoint: {x_goal}, {y_goal}, {theta_goal}")
            if waypoint != current_position:
                self.navigate_to_waypoint(x_goal, y_goal, theta_goal)
                self.set_current_position(waypoint)

        self.get_logger().info("All waypoints reached.")
        self.mpi_ctrl.carStop()
        self.mpi_ctrl.close()

    def publish_status(self):  # Periodic status publisher
        msg = String()
        msg.data = f"Current Position: {current_position}"
        self.pub.publish(msg)
        self.get_logger().info(f"Published status: {msg.data}")

def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS2 communication
    navigator = WaypointNavigator(waypoint_file='waypoints.txt')  # Initialize the waypoint navigator
    rclpy.spin(navigator)  # Keep the node spinning

    # Cleanup when the node is done
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
