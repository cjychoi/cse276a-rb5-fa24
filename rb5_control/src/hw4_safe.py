# safe.py

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from mpi_control import MegaPiController
import heapq

def create_world(length, width):
    m = int(length / 0.1) + 1  # rows
    n = int(width / 0.1) + 1  # columns
    world_grid = np.zeros((m, n))

    # Adding world boundaries
    for i in range(m):
        world_grid[i][0] = 1
        world_grid[i][n - 1] = 1
    for j in range(n):
        world_grid[0][j] = 1
        world_grid[m - 1][j] = 1

    return world_grid, m, n

def set_obstacle(world_grid, startx, starty, obLen, obWid):
    l = int(obLen / 0.1) + 1
    w = int(obWid / 0.1) + 1
    world_grid[startx: startx + l, starty: starty + w] = 1
    return world_grid

def heuristic(goal, cur):
    d_diagonal = min(abs(goal[0] - cur[0]), abs(goal[1] - cur[1]))
    d_straight = abs(goal[0] - cur[0]) + abs(goal[1] - cur[1])
    return math.sqrt(2) * d_diagonal + (d_straight - 2 * d_diagonal)

def move_cost(next, cur):
    return math.sqrt((next[0] - cur[0]) ** 2 + (next[1] - cur[1]) ** 2)

def safety_cost(cur, m, n, world_grid):
    d_obstacle = float('inf')
    for i in range(m):
        for j in range(n):
            if world_grid[i, j] == 1:  # Check for obstacles
                d_obstacle = min(d_obstacle, move_cost((i, j), cur))
    return 1.0 / (d_obstacle + 1e-5)  # Inverse safety: lower cost if farther from obstacles

def equals(goal, cur):
    return goal[0] == cur[0] and goal[1] == cur[1]

def safety_cost(cur, m, n, world_grid):
    d_obstacle = float('inf')
    for i in range(m):
        for j in range(n):
            if world_grid[i, j] == 1:  # Check for obstacles
                d_obstacle = min(d_obstacle, move_cost((i, j), cur))
    return 1.0 / (d_obstacle + 1e-5)  # Inverse safety: lower cost if farther from obstacles

def greedy_search(start, goal, m, n, world_grid, mw, hw, sw, mode='safety'):
    directions = [[-1, 0], [0, 1], [-1, 1], [1, 1], [1, 0], [0, -1], [-1, -1], [1, -1]]  # Diagonal + Cardinal
    cur = start
    path = [cur]

    while not equals(cur, goal):
        min_cost = float('inf')
        tmp = cur

        for direction in directions:
            nextX, nextY = cur[0] + direction[0], cur[1] + direction[1]
            if nextX < 0 or nextX >= m or nextY < 0 or nextY >= n:
                continue
            if world_grid[nextX, nextY] == 1:  # Skip obstacles
                continue

            next = (nextX, nextY)
            estimated_cost = mw * move_cost(next, cur) + hw * heuristic(goal, next) + sw * safety_cost(next, m, n, world_grid)

            if estimated_cost < min_cost:
                min_cost = estimated_cost
                tmp = next

        # Prevent invalid moves
        if tmp == cur:
            print("Stuck! No valid path found.")
            break

        cur = tmp
        path.append(cur)

    return path

def generate_safety_path():
    world_grid, m, n = create_world(length=3, width=3)

    # Set obstacles
    obstacle_positions = [
        (0, 0, 0.3, 0.3),  # Borders as obstacles
        (27, 0, 0.3, 0.3),
        (0, 27, 0.3, 0.3),
        (27, 27, 0.3, 0.3)
    ]
    center_obstacle = [(1.2, 1.2), (1.8, 1.8)]
    world_grid = set_obstacle(world_grid, int(center_obstacle[0][0] / 0.1), int(center_obstacle[0][1] / 0.1), 
                              center_obstacle[1][0] - center_obstacle[0][0], 
                              center_obstacle[1][1] - center_obstacle[0][1])

    for x, y, l, w in obstacle_positions:
        world_grid = set_obstacle(world_grid, int(x / 0.1), int(y / 0.1), l, w)

    # Define start and goal
    start_point = (24, 3)  # Scaled to grid index
    target_point = (6, 24)  # Scaled to grid index

    # Safety path planning
    safety_path = greedy_search(start_point, target_point, m, n, world_grid, mw=1, hw=2, sw=100, mode='safety')

    # Convert to real-world coordinates and round to 1 decimal point
    path = [(round(p[0] * 0.1, 1), round(p[1] * 0.1, 1)) for p in safety_path]

    # Calculate shortest distance to obstacle along the path
    shortest_distance = float('inf')
    for point in safety_path:
        # Calculate the distance from the point to the closest obstacle
        d_to_obstacle = safety_cost(point, m, n, world_grid)
        # Keep track of the shortest distance
        if d_to_obstacle < shortest_distance:
            shortest_distance = d_to_obstacle

    # Print the shortest distance to any obstacle
    print(f"Shortest distance to an obstacle: {round(1 / shortest_distance, 2)} meters")  # Inverse of safety cost
    
    
    return world_grid, path, center_obstacle

# Plot grid and path
def plot_path(grid, path, center_obstacle, resolution=0.1):
    plt.figure(figsize=(8, 8))
    # Plot the grid
    # plt.imshow(grid.T, cmap="Greys", origin="lower", extent=[0, 3, 0, 3])
    
    # Plot the center obstacle as a blue rectangle
    plt.gca().add_patch(
        Rectangle(center_obstacle[0], center_obstacle[1][0] - center_obstacle[0][0], center_obstacle[1][1] - center_obstacle[0][1], 
                  color="blue", alpha=0.5, label="Center Obstacle")
    )
    
    # Plot the path
    if path:
        px, py = zip(*path)
        plt.plot(px, py, color="red", lw=2, label="Safety Path")
    
    # Plot start and goal points
    start, goal = path[0], path[-1]
    plt.scatter([start[0]], [start[1]], color="green", label="Start", zorder=5)
    plt.scatter([goal[0]], [goal[1]], color="purple", label="Goal", zorder=5)
    
    # Add labels, grid, and legend
    plt.legend()
    plt.grid()
    plt.title("Pathfinding in 3x3 Grid with Safety Optimization")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.xlim(0, 3)  # Adjust limits as per your grid size
    plt.ylim(0, 3)  # Adjust limits as per your grid size
    plt.show()

def rotate_to_angle(angle_diff, rad_per_sec, k_w):    # rotate robot to rotational goal by amount of time
    # Calculate rotation time based on angle difference
    rotation_time = abs(angle_diff) / rad_per_sec
    mpi_ctrl.carRotate(-k_w if angle_diff > 0 else k_w)
    time.sleep(rotation_time)
    mpi_ctrl.carStop()

def move_straight(distance, dist_per_sec, k_v):        # move robot straight by amount of time
    # Calculate movement time based on distance
    movement_time = abs(distance) / (dist_per_sec / 100)  # Convert cm/s to m/s
    mpi_ctrl.carStraight(k_v)
    time.sleep(movement_time)
    mpi_ctrl.carStop()
    # time.sleep(1)

if __name__ == '__main__':

    # store start time 
    begin = time.time()
    
    # Initialize the MegaPiController
    mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
    time.sleep(1)  # Allow some time for the connection to establish
  
    # Generate safety path
    world_grid, safety_path, center_obstacle = generate_safety_path()

    # Print safety path
    print("Safety Path (meters):")
    for waypoint in safety_path:
        print(waypoint)

    move_list = []
        
    
    # Assume the robot is initially facing directly up (90 degrees)
    initial_orientation = np.radians(90)  # Convert 90 degrees to radians
    
    # Get the start and the first waypoint
    start_waypoint = safety_path[0]
    first_waypoint = safety_path[1]
    
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
    move_list.append((0, round(np.radians(angle_of_rotation), 2)))
    
    prev_angle = None
    
    # Calculate the total moving distance
    total_distance = 0.0
    for i in range(1, len(safety_path)):
        prev = safety_path[i - 1]
        curr = safety_path[i]
    
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
        
        # Compute Euclidean distance between consecutive waypoints
        distance = np.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
        print(f"Moving Distance to Next Waypoint: {round(distance, 2)} meters")
        total_distance += distance

        move_list.append((distance, round(np.radians(angle_diff), 2)))
    
    # Print total distance
    print(f"Total Moving Distance: {round(total_distance, 2)} meters")

    print(move_list)

    # Control parameters prepared from calibration
    k_v = 28  # Speed for straight movement
    k_w = 50  # Speed for rotational movement
    dist_per_sec = 10 / 1  # 10 cm per 1 second at speed 30 for straight movement   
    rad_per_sec = math.pi / 2  # Pi radians per 2 seconds at speed 55 for rotational movement
    tolerance = 0.1  # Distance tolerance to waypoint (meters)

    rotate_to_angle(move_list[0][1], rad_per_sec, k_w)

    for move in move_list[1:]:
    
        dist, rot = move
        print(dist, rot)

        if abs(rot) != 0.0:
            if angle > 180:
                angle = angle - 360
            if angle < -180:
                angle = 360 - angle
            rotate_to_angle(rot, rad_per_sec, k_w)
        move_straight(dist, dist_per_sec, k_v)

    # Plot the path
    plot_path(world_grid, safety_path, center_obstacle)

    # store end time 
    end = time.time() 
     
    # total time taken 
    print(f"Total runtime of the program is {end - begin}") 
