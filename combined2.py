# imports
import time
import math
import rclpy
from rclpy.node import Node
from mpi_control import MegaPiController

current_position = [0.0, 0.0, 0.0]        # initialize global array to hold current position

class WaypointNavigator(Node):    #class to hold all functions
    global current_position    # declare current_position as global array
    def __init__(self, waypoint_file):
        super().__init__('navigator')

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

    def load_waypoints(self, filename):        # load waypoints from file
        waypoints = []
        with open(filename, 'r') as f:        # open file, read waypoints line-by-line, put into array of arrays
            for line in f.readlines():
                x, y, theta = map(float, line.strip().split(','))      
                waypoints.append((x, y, theta))
        print('waypoints: ', waypoints)
        return waypoints

    def calculate_distance(self, x1, y1, x2, y2):        # calculate distance of goal from current location of robot
        return (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))/2

    def calculate_angle(self, x1, y1, x2, y2):        # calculate angular distance of goal from current rotation of robot
        if (x1 == x2) and (y1 != y2):        # if goal position is on x-axis of robot
            if (y2 > y1):
                return 0
            else:
                return math.pi
        elif (y1 == y2) and (x1 != x2):        # if goal position is on y-axis of robot
            if (x1 > x2):
                return math.pi / 2
            else:
                return -(math.pi / 2)
        else:                                # if x and y movement necessary to get to goal
            return math.atan2(y2 - y1, x2 - x1) - math.pi/2

    def reached_waypoint(self, x_goal, y_goal):    # when the robot has moved or rotated, check if rotation or straight movement is needed to reach goal position
        x, y, _ = self.get_current_position()    # get the robot's current position
        distance = self.calculate_distance(x, y, x_goal, y_goal)       
        return distance < self.tolerance

    def get_current_position(self):    # get the robot's current position from global array
        return current_position[0], current_position[1], current_position[2]  # x, y, theta

    def set_current_position(self, waypoint):    # store the robot's current position in global array
        current_position[0] = waypoint[0]
        current_position[1] = waypoint[1]
        current_position[2] = waypoint[2]
        print('set current position: ', waypoint)

    def rotate_to_angle(self, angle_diff):    # rotate robot to rotational goal by amount of time
        # Calculate rotation time based on angle difference
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

    def move_straight(self, distance):        # move robot straight by amount of time
        # Calculate movement time based on distance
        movement_time = abs(distance) / (self.dist_per_sec / 100)  # Convert cm/s to m/s
        self.mpi_ctrl.carStraight(self.k_v)
        time.sleep(movement_time)
        self.mpi_ctrl.carStop()

    def navigate_to_waypoint(self, x_goal, y_goal, theta_goal):        #m move robot to x,y,theta goal position
        print('navigate to waypoint')
        while not self.reached_waypoint(x_goal, y_goal):        # while not in position, checked by comparing goal position and current position
            x, y, theta = self.get_current_position()

            # Calculate distance and angle to the goal
            distance = self.calculate_distance(x, y, x_goal, y_goal)   
            print('distance: ', distance)
            angle_to_goal = self.calculate_angle(x, y, x_goal, y_goal) 
            angle_diff = angle_to_goal - theta
            print('angle_to_goal: ', angle_to_goal, ' | theta: ', theta)
            print('angle_diff: ', angle_diff)

            if abs(angle_diff) > 0.1:  # Rotate first if not facing the goal
                self.rotate_to_angle(angle_diff)
                self.set_current_position([x, y, angle_to_goal])       
            else:  # Move straight if facing the goal
                self.move_straight(distance)
                # rotate to theta goal position
                angle_to_goal = self.calculate_angle(x, y, x_goal, y_goal)
                angle_diff = theta_goal - angle_to_goal
                self.rotate_to_angle(angle_diff)
                self.set_current_position([x_goal, y_goal, theta_goal])

    def start_navigation(self):    # start movement of robot to waypoints
        for waypoint in self.waypoints:    # for each waypoint, set x,y,theta goal and travel to goal position
            x_goal, y_goal, theta_goal = waypoint
            print(f"Navigating to waypoint: {x_goal}, {y_goal}, {theta_goal}")
            if (waypoint != current_position):
                self.navigate_to_waypoint(x_goal, y_goal, theta_goal)  
                self.set_current_position(waypoint)

        print("All waypoints reached.")
        self.mpi_ctrl.carStop()
        self.mpi_ctrl.close()

if __name__ == "__main__":
    rclpy.init(args=None)
    
    # Assuming waypoints.txt is the file with the list of waypoints    
    navigator = WaypointNavigator(waypoint_file='waypoints.txt')       # load list of waypoints into program
    navigator.start_navigation()                                       # start movement

    rclpy.spin(navigator)

    navigator.destroy_node()
    rclpy.shutdown()
    