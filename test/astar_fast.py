# imports
import time
import math
from mpi_control import MegaPiController
import numpy as np
import matplotlib.pyplot as plt
import heapq

# Convert coordinates to grid indices
def to_grid(x, y, resolution=0.1, grid_size=3.0):
    max_index = int(grid_size / resolution) - 1
    grid_x = max(0, min(int(x / resolution), max_index))
    grid_y = max(0, min(int(y / resolution), max_index))
    return grid_x, grid_y

# Heuristic function (Euclidean distance)
def heuristic(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# A* Algorithm
def a_star(grid, start, goal):
    start = to_grid(*start)
    goal = to_grid(*goal)
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    # Define all possible moves, including diagonal and intermediate directions
    moves = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal directions
        (-1, -1), (-1, 1), (1, -1), (1, 1),  # Diagonal directions
        (-2, -1), (-2, 1), (2, -1), (2, 1),  # Intermediate angles
        (-1, -2), (-1, 2), (1, -2), (1, 2)
    ]

    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for dx, dy in moves:
            neighbor = (current[0] + dx, current[1] + dy)
            if (
                0 <= neighbor[0] < grid.shape[0] and
                0 <= neighbor[1] < grid.shape[1] and
                grid[neighbor[0], neighbor[1]] == 0
            ):
                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found

# Plot grid and path
def plot_path(grid, path, resolution=0.1):
    plt.figure(figsize=(8, 8))
    
    # Plot the path
    if path:
        px, py = zip(*path)
        plt.plot(np.array(px) * resolution, np.array(py) * resolution, color="red", lw=2, label="Path")
    
    # Plot obstacles
    plt.scatter(*zip(*obstacle_positions), c="blue", label="Landmarks")
    
    # Plot the center obstacle as a rectangle
    plt.gca().add_patch(plt.Rectangle((1.2, 1.2), 0.6, 0.6, color="blue", alpha=0.5, label="Center Obstacle"))
    
    # Plot start and goal points
    plt.scatter([start[0]], [start[1]], color="green", label="Start", zorder=5)
    plt.scatter([goal[0]], [goal[1]], color="purple", label="Goal", zorder=5)
    
    # Add labels, grid, and legend
    plt.legend()
    plt.grid()
    plt.title("Pathfinding in 3x3 Grid with Flexible Turns")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.xlim(0, 3)  # Adjust limits as per your grid size
    plt.ylim(0, 3)  # Adjust limits as per your grid size
    plt.show()

# Convert grid indices back to coordinates
def to_coordinates(grid_x, grid_y, resolution=0.1):
    return grid_x * resolution, grid_y * resolution

# Initialize global current position
current_position = [0.0, 0.0, 0.0]

class WaypointNavigator:
    def __init__(self, waypoints):
        global current_position
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)  # Allow some time for the connection to establish

        self.waypoints = waypoints
        self.current_waypoint_idx = 0

        # Control parameters (calibrated)
        self.k_v = 30  # Speed for straight movement
        self.k_w = 58  # Speed for rotational movement
        self.dist_per_sec = 10 / 1  # 10 cm per second at speed 30 for straight movement   
        self.rad_per_sec = math.pi / 2  # Pi radians per 2 seconds at speed 55 for rotational movement
        self.tolerance = 0.1  # Distance tolerance to waypoint (meters)

    def calculate_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_angle(self, x1, y1, x2, y2):
        return math.atan2(y2 - y1, x2 - x1)

    def reached_waypoint(self, x_goal, y_goal):
        x, y, _ = self.get_current_position()
        distance = self.calculate_distance(x, y, x_goal, y_goal)
        return distance < self.tolerance

    def get_current_position(self):
        return current_position[0], current_position[1], current_position[2]

    def set_current_position(self, waypoint):
        current_position[0], current_position[1], current_position[2] = waypoint
        print('Current position set to:', waypoint)

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
        print('Navigating to waypoint...')
        while not self.reached_waypoint(x_goal, y_goal):
            x, y, theta = self.get_current_position()

            # Calculate distance and angle to the goal
            distance = self.calculate_distance(x, y, x_goal, y_goal)
            angle_to_goal = self.calculate_angle(x, y, x_goal, y_goal)
            angle_diff = angle_to_goal - theta

            if abs(angle_diff) > 0.1:
                self.rotate_to_angle(angle_diff)
                self.set_current_position([x, y, angle_to_goal])
            else:
                self.move_straight(distance)
                self.rotate_to_angle(theta_goal - angle_to_goal)
                self.set_current_position([x_goal, y_goal, theta_goal])

    def start_navigation(self):
        for waypoint in self.waypoints:
            x_goal, y_goal, theta_goal = waypoint
            print(f"Navigating to waypoint: {x_goal}, {y_goal}, {theta_goal}")
            self.navigate_to_waypoint(x_goal, y_goal, theta_goal)

        print("All waypoints reached.")
        self.mpi_ctrl.carStop()
        self.mpi_ctrl.close()

if __name__ == "__main__":
    # Define the grid and obstacles
    grid_size = 3.0
    obstacle_positions = [
        (0.3, 0), (2.7, 0), (0, 0.3), (3, 0.3), 
        (0, 2.7), (0.3, 3), (3, 2.7), (2.7, 3)
    ]
    center_obstacle = [(1.2, 1.2), (1.8, 1.8)]
    start = (2.4, 0.3)
    goal = (0.6, 2.4)

    # Generate grid and mark obstacles
    resolution = 0.1
    grid = np.zeros((int(grid_size / resolution), int(grid_size / resolution)))
    for x, y in obstacle_positions:
        gx, gy = to_grid(x, y)
        grid[gx, gy] = 1
    for x, y in np.ndindex(grid.shape):
        if 1.2 <= x * resolution <= 1.8 and 1.2 <= y * resolution <= 1.8:
            grid[x, y] = 1

    # Find path
    path = a_star(grid, start, goal)
    print(f"Found path: {path}")
    plot_path(grid, path)

    # Convert path back to coordinates
    if path:
        waypoints = [to_coordinates(gx, gy) + (0,) for gx, gy in path]  # Include theta=0 for each waypoint
        navigator = WaypointNavigator(waypoints)
        navigator.start_navigation()
