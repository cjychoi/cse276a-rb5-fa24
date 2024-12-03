import matplotlib.pyplot as plt
import numpy as np

    # Updated coordinates
    # object_coords = [
    #     (0, -0.2, "bottle"),
    #     (2.1, 0, "stop sign"),
    #     (2.1, 1.5, "potted plant"),
    #     (1.4, -0.3, "teddy bear"),
    #     (-0.7, 0.2, "umbrella"),
    #     (-0.4, 1.3, "keyboard"),
    #     (-0.2, 1.9, "suitcase"),
    #     (1.3, 1.6, "laptop")
    # ]

def make_square(object_coords):

    # Extract x and y coordinates
    x_coords = np.array([coord[0] for coord in object_coords])
    y_coords = np.array([coord[1] for coord in object_coords])

    # Calculate the center point (centroid)
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)

    # Function to calculate angle relative to the center point
    def calculate_angle(x, y, center_x, center_y):
        dx = x - center_x
        dy = y - center_y
        return np.degrees(np.arctan2(dy, dx)) % 360

    # Categorize points into quadrants based on their angles
    quadrant_1 = []  # Top boundary
    quadrant_2 = []  # Left boundary
    quadrant_3 = []  # Bottom boundary
    quadrant_4 = []  # Right boundary

    for x, y, label in object_coords:
        angle = calculate_angle(x, y, center_x, center_y)
        if 0 <= angle < 90:  # Quadrant 1
            quadrant_1.append((x, y))
        elif 90 <= angle < 180:  # Quadrant 2
            quadrant_2.append((x, y))
        elif 180 <= angle < 270:  # Quadrant 3
            quadrant_3.append((x, y))
        else:  # Quadrant 4
            quadrant_4.append((x, y))

    # Calculate boundaries
    top_boundary = min(y for x, y in quadrant_1) if quadrant_1 else center_y
    left_boundary = max(x for x, y in quadrant_2) if quadrant_2 else center_x
    bottom_boundary = max(y for x, y in quadrant_3) if quadrant_3 else center_y
    right_boundary = min(x for x, y in quadrant_4) if quadrant_4 else center_x

    # Calculate the side of the largest square that fits
    square_side = min(top_boundary - bottom_boundary, right_boundary - left_boundary)

    # Calculate square coordinates
    square_start_x = center_x - square_side / 2
    square_start_y = center_y - square_side / 2
    square_end_x = square_start_x + square_side
    square_end_y = square_start_y + square_side

    # Plot the points and the adjusted square
    plt.figure(figsize=(8, 8))
    for x, y, label in object_coords:
        plt.scatter(x, y, label=label)

    # Draw the square
    square_x = [square_start_x, square_end_x, square_end_x, square_start_x, square_start_x]
    square_y = [square_start_y, square_start_y, square_end_y, square_end_y, square_start_y]
    plt.plot(square_x, square_y, 'r-', label='Largest Square')

    # Mark the center point
    plt.scatter(center_x, center_y, color='black', label='Center Point (Centroid)')

    # Plot quadrant lines
    plt.plot([center_x, center_x + 2], [center_y, center_y + 2], 'k--')  # 45 degrees
    plt.plot([center_x, center_x - 2], [center_y, center_y + 2], 'k--')  # 135 degrees
    plt.plot([center_x, center_x - 2], [center_y, center_y - 2], 'k--')  # 225 degrees
    plt.plot([center_x, center_x + 2], [center_y, center_y - 2], 'k--')  # 315 degrees

    # Plot aesthetics
    # plt.legend()
    plt.title("Largest Square Fit Using Quadrants")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True)
    plt.axis('equal')

    # Save the plot as a PNG file
    plt.savefig("square.png")
    plt.show()

    # Output dimensions
    print(f"Largest square dimensions: {square_side}m x {square_side}m")
    print(f"Square center point: ({center_x}, {center_y})")
    print(f"Square coordinates: ({square_start_x}, {square_start_y}) to ({square_end_x}, {square_end_y})")