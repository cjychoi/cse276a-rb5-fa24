import time
import math
from mpi_control import MegaPiController

current_position = [0.0, 0.0, 0.0]

class WaypointNavigator:
    global current_position
    def __init__(self, waypoint_file):

        # Initialize the MegaPiController
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)  # Allow some time for the connection to establish

        # Load waypoints from a file
        self.waypoints = self.load_waypoints(waypoint_file)
        self.current_waypoint_idx = 0

        # Control parameters
        self.k_v = 30  # Speed for straight movement
        self.k_w = 38  # Speed for rotation
        self.dist_per_sec = 10 / 1  # 10 cm per 1 second at speed 30   
        self.rad_per_sec = math.pi / 3.6  # Pi radians per 3.6 seconds at speed 38
        self.tolerance = 0.1  # Distance tolerance to waypoint (meters)

    def load_waypoints(self, filename):
        waypoints = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                x, y, theta = map(float, line.strip().split(','))      
                waypoints.append((x, y, theta))
        return waypoints

    def calculate_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_angle(self, x1, y1, x2, y2):
        return math.atan2(y2 - y1, x2 - x1)

    def reached_waypoint(self, x_goal, y_goal):
        x, y, _ = self.get_current_position()  # Placeholder function to get the robot's position
        distance = self.calculate_distance(x, y, x_goal, y_goal)       
        return distance < self.tolerance

    def get_current_position(self):
        # Placeholder: Replace this with actual robot position retrieval logic
        return current_position[0], current_position[1], current_position[2]  # x, y, theta

    def set_current_position(self, waypoint):
        current_position = waypoint

    def rotate_to_angle(self, angle_diff):
        # Calculate rotation time based on angle difference
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

    def move_straight(self, distance):
        # Calculate movement time based on distance
        movement_time = abs(distance) / (self.dist_per_sec / 100)  # Convert cm/s to m/s
        self.mpi_ctrl.carStraight(self.k_v)
        time.sleep(movement_time)
        self.mpi_ctrl.carStop()

    def navigate_to_waypoint(self, x_goal, y_goal, theta_goal):        
        print(4)
        while not self.reached_waypoint(x_goal, y_goal):
            x, y, theta = self.get_current_position()

            # Calculate distance and angle to the goal
            distance = self.calculate_distance(x, y, x_goal, y_goal)   
            print('distance: ', distance)
            angle_to_goal = self.calculate_angle(x, y, x_goal, y_goal) 
            angle_diff = angle_to_goal - theta
            print('angle_diff: ', angle_diff)

            if abs(angle_diff) > 0.1:  # Rotate first if not facing the goal
                print('5a')
                self.rotate_to_angle(angle_diff)
            else:  # Move straight if facing the goal
                print('5b')
                self.move_straight(distance)

        self.mpi_ctrl.carStop()

    def start_navigation(self):
        print(2)
        for waypoint in self.waypoints:
            x_goal, y_goal, theta_goal = waypoint
            print(f"Navigating to waypoint: {x_goal}, {y_goal}, {theta_goal}")
            print(3)
            if (waypoint != current_position):
                self.navigate_to_waypoint(x_goal, y_goal, theta_goal)  
                self.set_current_position(waypoint)

        print("All waypoints reached.")
        self.mpi_ctrl.carStop()
        self.mpi_ctrl.close()

if __name__ == "__main__":
    # Assuming waypoints.txt is the file with the list of waypoints    
    navigator = WaypointNavigator(waypoint_file='waypoints.txt')       
    print(1)
    navigator.start_navigation()
