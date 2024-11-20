import numpy as np
import matplotlib.pyplot as plt

# Configuration
grid_size = 3.0
resolution = 0.1
start = np.array([2.4, 0.3])
goal = np.array([0.6, 2.4])
obstacle_positions = [
    (0.3, 0), (2.7, 0), (0, 0.3), (3, 0.3), 
    (0, 2.7), (0.3, 3), (3, 2.7), (2.7, 3)
]
center_obstacle = [(1.2, 1.2), (1.8, 1.8)]  # Define as top-left and bottom-right corners

# Potential field parameters
repulsive_strength = 100.0
repulsive_radius = 0.5
attractive_strength = 5.0

def to_grid(x, y, resolution=0.1, grid_size=3.0):
    max_index = int(grid_size / resolution) - 1
    grid_x = max(0, min(int(x / resolution), max_index))
    grid_y = max(0, min(int(y / resolution), max_index))
    return grid_x, grid_y

# Compute potential fields
def compute_potential_field(grid_size, resolution, start, goal, obstacle_positions, center_obstacle):
    x = np.arange(0, grid_size, resolution)
    y = np.arange(0, grid_size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Attractive potential
    U_att = 0.5 * attractive_strength * ((X - goal[0]) ** 2 + (Y - goal[1]) ** 2)
    
    # Repulsive potential
    U_rep = np.zeros_like(U_att)
    for obs in obstacle_positions:
        dist = np.sqrt((X - obs[0]) ** 2 + (Y - obs[1]) ** 2)
        U_rep += np.where(
            dist <= repulsive_radius,
            0.5 * repulsive_strength * ((1.0 / dist) - (1.0 / repulsive_radius)) ** 2,
            0
        )
    
    # Add center obstacle
    cx_min, cy_min = center_obstacle[0]
    cx_max, cy_max = center_obstacle[1]
    for cx in np.arange(cx_min, cx_max, resolution):
        for cy in np.arange(cy_min, cy_max, resolution):
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            U_rep += np.where(
                dist <= repulsive_radius,
                0.5 * repulsive_strength * ((1.0 / dist) - (1.0 / repulsive_radius)) ** 2,
                0
            )
    
    U_total = U_att + U_rep
    return U_total, X, Y

# Gradient descent to follow the potential field
def gradient_descent(U, start, goal, resolution, step_size=0.1, max_iters=500):
    path = [start]
    pos = start.copy()
    goal_grid = to_grid(*goal, resolution)

    for _ in range(max_iters):
        grid_pos = to_grid(pos[0], pos[1], resolution)
        if grid_pos == goal_grid:
            break

        i, j = grid_pos
        if 1 <= i < U.shape[0] - 1 and 1 <= j < U.shape[1] - 1:
            grad_x = (U[i + 1, j] - U[i - 1, j]) / (2 * resolution)
            grad_y = (U[i, j + 1] - U[i, j - 1]) / (2 * resolution)
        else:
            grad_x, grad_y = 0, 0  # Out of bounds check

        grad = np.array([grad_x, grad_y])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad /= grad_norm  # Normalize the gradient to unit vector

        pos -= step_size * grad  # Move in the direction of negative gradient
        path.append(pos.copy())

        # Stop if sufficiently close to goal
        if np.linalg.norm(pos - goal) < resolution:
            break

    return path

# Compute potential fields and follow the gradient
U_total, X, Y = compute_potential_field(grid_size, resolution, start, goal, obstacle_positions, center_obstacle)
path = gradient_descent(U_total, start, goal, resolution)

# Plot results
plt.figure(figsize=(8, 8))
plt.contourf(X, Y, U_total, levels=50, cmap="coolwarm")
px, py = zip(*path)
plt.plot(px, py, color="yellow", lw=2, label="Safe Path")
plt.scatter(*zip(*obstacle_positions), c="blue", label="Landmarks")
plt.gca().add_patch(plt.Rectangle(center_obstacle[0], 0.6, 0.6, color="blue", alpha=0.5, label="Center Obstacle"))
plt.scatter([start[0]], [start[1]], color="green", label="Start", zorder=5)
plt.scatter([goal[0]], [goal[1]], color="purple", label="Goal", zorder=5)
plt.legend()
plt.title("Maximum Safety Path (Potential Fields)")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.grid()
plt.show()