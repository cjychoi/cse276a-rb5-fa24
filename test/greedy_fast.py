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

# Greedy Search Algorithm
def greedy_search(grid, start, goal):
    start = to_grid(*start)
    goal = to_grid(*goal)
    open_set = [(heuristic(start, goal), start)]
    came_from = {}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        visited.add(current)

        neighbors = [
            (current[0] + dx, current[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]

        for neighbor in neighbors:
            if (
                0 <= neighbor[0] < grid.shape[0] and
                0 <= neighbor[1] < grid.shape[1] and
                grid[neighbor[0], neighbor[1]] == 0 and
                neighbor not in visited
            ):
                visited.add(neighbor)
                came_from[neighbor] = current
                heapq.heappush(open_set, (heuristic(neighbor, goal), neighbor))

    return None  # No path found

# Plot the path
def plot_path(grid, path, start, goal, obstacles, center_obstacle, resolution=0.1):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.T, cmap="Greys", origin="lower")
    if path:
        px, py = zip(*path)
        plt.plot(np.array(px) * resolution, np.array(py) * resolution, color="red", lw=2, label="Path")
    plt.scatter(*zip(*obstacles), c="blue", label="Landmarks")
    plt.gca().add_patch(plt.Rectangle(center_obstacle[0], 0.6, 0.6, color="blue", alpha=0.5, label="Center Obstacle"))
    plt.scatter([start[0]], [start[1]], color="green", label="Start", zorder=5)
    plt.scatter([goal[0]], [goal[1]], color="purple", label="Goal", zorder=5)
    plt.legend()
    plt.grid()
    plt.title("Shortest Path (Greedy Search)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.show()

# Configuration
grid_size = 3.0
resolution = 0.1
start = (2.4, 0.3)
goal = (0.6, 2.4)
obstacle_positions = [
    (0.3, 0), (2.7, 0), (0, 0.3), (3, 0.3), 
    (0, 2.7), (0.3, 3), (3, 2.7), (2.7, 3)
]
center_obstacle = [(1.2, 1.2), (1.8, 1.8)]  # Define as top-left and bottom-right corners

# Generate grid
grid = np.zeros((int(grid_size / resolution), int(grid_size / resolution)))
for x, y in obstacle_positions:
    gx, gy = to_grid(x, y, resolution)
    grid[gx, gy] = 1
cx1, cy1 = to_grid(*center_obstacle[0], resolution)
cx2, cy2 = to_grid(*center_obstacle[1], resolution)
grid[cx1:cx2+1, cy1:cy2+1] = 1

# Find shortest path
path = greedy_search(grid, start, goal)

# Plot the result
plot_path(grid, path, start, goal, obstacle_positions, center_obstacle)