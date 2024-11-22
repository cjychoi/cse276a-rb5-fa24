# safe.py

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

if __name__ == '__main__':
    # Generate safety path
    world_grid, safety_path, center_obstacle = generate_safety_path()

    # Print safety path
    print("Safety Path (meters):")
    for waypoint in safety_path:
        print(waypoint)

    # Plot the path
    plot_path(world_grid, safety_path, center_obstacle)