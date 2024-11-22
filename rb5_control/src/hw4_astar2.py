# imports
import time
import math
from mpi_control import MegaPiController
import numpy as np
import matplotlib.pyplot as plt
import heapq

# Define the grid and obstacles
grid_size = 3.0
obstacle_positions = [
    (0.3, 0), (2.7, 0), (0, 0.3), (3, 0.3), 
    (0, 2.7), (0.3, 3), (3, 2.7), (2.7, 3)
]
center_obstacle = [(1.2, 1.2), (1.8, 1.8)]  # Define as top-left and bottom-right corners
start = (2.4, 0.3)
goal = (0.6, 2.4)

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

# Generate grid and mark obstacles
resolution = 0.1
grid = np.zeros((int(grid_size / resolution), int(grid_size / resolution)))
for x, y in obstacle_positions:
    gx, gy = to_grid(x, y, resolution)
    grid[gx, gy] = 1
cx1, cy1 = to_grid(*center_obstacle[0], resolution)
cx2, cy2 = to_grid(*center_obstacle[1], resolution)
grid[cx1:cx2+1, cy1:cy2+1] = 1

# Find shortest path
path = a_star(grid, start, goal)

# Plot grid and path
def plot_path(grid, path, resolution=0.1):
    plt.figure(figsize=(8, 8))
    # plt.imshow(grid.T, cmap="Greys", origin="lower")
    
    # Plot the path
    if path:
        px, py = zip(*path)
        plt.plot(np.array(px) * resolution, np.array(py) * resolution, color="red", lw=2, label="Path")
    
    # Plot obstacles
    plt.scatter(*zip(*obstacle_positions), c="blue", label="Landmarks")
    
    # Plot the center obstacle as a rectangle
    # plt.gca().add_patch(
    #     plt.Rectangle(center_obstacle[0], 0.6, 0.6, color="blue", alpha=0.5, label="Center Obstacle")
    # )
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

# Convert grid indices back to coordinates for waypoints
waypoints = [tuple(round(coord, 1) for coord in to_coordinates(px, py, resolution)) for px, py in path]

# Print waypoints
print("Waypoints (in meters):")
for waypoint in waypoints:
    print(waypoint)

waypoint_list = []
move_list = []

# Assume the robot is initially facing directly up (90 degrees)
initial_orientation = np.radians(90)  # Convert 90 degrees to radians

# Get the start and the first waypoint
start_waypoint = waypoints[0]
first_waypoint = waypoints[1]

# Calculate the direction vector from start to the first waypoint
direction_vector = (first_waypoint[0] - start_waypoint[0], first_waypoint[1] - start_waypoint[1])

# Calculate the angle of the direction vector (in radians)
angle_to_first_waypoint = np.arctan2(direction_vector[1], direction_vector[0])

# Calculate the angle of rotation needed to face the first waypoint
# Rotation = angle_to_first_waypoint - initial_orientation
angle_of_rotation = np.degrees(angle_to_first_waypoint - initial_orientation)

# Normalize the angle to [-180, 180] range
angle_of_rotation = (angle_of_rotation + 180) % 360 - 180

# Output the required angle of rotation
print(f"Angle of rotation to face the first waypoint: {round(angle_of_rotation, 2)} degrees")
waypoint_list.append((start_waypoint[0], start_waypoint[1], round(np.radians(angle_of_rotation), 2)))
move_list.append((0, round(np.radians(angle_of_rotation), 2)))

prev_angle = None

# Calculate the total moving distance
total_distance = 0.0
for i in range(1, len(waypoints)):
    prev = waypoints[i - 1]
    curr = waypoints[i]

    # Calculate direction vector between consecutive waypoints
    direction_vector = (curr[0] - prev[0], curr[1] - prev[1])
    # Calculate angle of the direction vector (in radians)
    angle = np.arctan2(direction_vector[1], direction_vector[0])  # Angle in radians
    # If there's a previous angle, compute the rotation needed
    if prev_angle is not None:
        angle_diff = np.degrees(angle - prev_angle)
        print(f"Rotation at this waypoint: {round(angle_diff, 2)} degrees")
        print(f"Angle at this waypoint: {round(np.degrees(angle), 2)} degrees")
    else:
        angle_diff = 0.0
    prev_angle = angle
    waypoint_list.append((waypoints[i][0], waypoints[i][1], round(np.radians(angle), 2)))
    
    # Compute Euclidean distance between consecutive waypoints
    distance = np.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
    print(f"Moving Distance to Next Waypoint: {round(distance, 2)} meters")
    total_distance += distance

    move_list.append((distance, round(np.radians(angle_diff), 2)))

# Print total distance
print(f"Total Moving Distance: {round(total_distance, 2)} meters")

print(waypoint_list)
print(move_list)

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
    time.sleep(2)

# Control parameters prepared from calibration
self.k_v = 30  # Speed for straight movement
self.k_w = 58  # Speed for rotational movement
self.dist_per_sec = 10 / 1  # 10 cm per 1 second at speed 30 for straight movement   
self.rad_per_sec = math.pi / 2  # Pi radians per 2 seconds at speed 55 for rotational movement
self.tolerance = 0.1  # Distance tolerance to waypoint (meters)

for move in move_list:
    dist, rot = move
    print(dist, rot)
    
    move_straight(dist)
    rotate_to_angle(rot)


plot_path(grid, path)
