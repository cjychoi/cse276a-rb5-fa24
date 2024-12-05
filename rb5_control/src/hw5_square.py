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

import matplotlib.pyplot as plt
import numpy as np

def make_square(object_coords, robot_width=0.17, robot_height=0.20, start_x=0, start_y=0):
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
        if 45 <= angle < 135:  # Quadrant 1
            quadrant_1.append((x, y))
        elif 135 <= angle < 225:  # Quadrant 2
            quadrant_2.append((x, y))
        elif 225 <= angle < 315:  # Quadrant 3
            quadrant_3.append((x, y))
        else:  # Quadrant 4
            quadrant_4.append((x, y))

    # Define margins for each constraint
    margin = 0.2

    # Calculate boundaries
    top_boundary = (min(y for x, y in quadrant_1) if quadrant_1 else center_y) 
    left_boundary = (max(x for x, y in quadrant_2)if quadrant_2 else center_x) 
    bottom_boundary = (max(y for x, y in quadrant_3) if quadrant_3 else center_y) 
    right_boundary = (min(x for x, y in quadrant_4) if quadrant_4 else center_x) 

    # Calculate the side of the largest square that fits
    square_side = min(top_boundary - bottom_boundary, right_boundary - left_boundary) - margin 

    # Calculate square coordinates
    square_start_x = center_x - square_side / 2
    square_start_y = center_y - square_side / 2
    square_end_x = square_start_x + square_side
    square_end_y = square_start_y + square_side

    # Find the nearest corner of the square to the start point (0, 0)
    corners = [
        (square_start_x, square_start_y),
        (square_end_x, square_start_y),
        (square_start_x, square_end_y),
        (square_end_x, square_end_y),
    ]
    nearest_corner = min(corners, key=lambda c: np.hypot(c[0] - start_x, c[1] - start_y))

    # Define the robot's back-and-forth sweeping path (Roomba-like)
    robot_path = [(start_x, start_y), nearest_corner]  # Start at (0,0) and move to nearest corner
    current_x, current_y = nearest_corner
    direction = 1  # 1 for right, -1 for left
    step_y = robot_height  # The y-step is based on the height of the robot


    while current_y <= square_end_y:
        # Move along the x-axis
        if direction == 1:  # Move right
            robot_path.append((square_end_x, current_y))
        else:  # Move left
            robot_path.append((square_start_x, current_y))
        
        # After reaching one side, move up by the width of the robot and reverse direction
        current_y += step_y
        if current_y <= square_end_y:  # Add only if within bounds
            robot_path.append((robot_path[-1][0], current_y))  # Move up in y direction
        
        # Switch direction for next horizontal movement
        direction *= -1

    # Print robot path points where the direction changes
    print("Robot direction change points:")
    for point in robot_path:
        print(point)

    # Plot the points, the adjusted square, and the robot's sweeping path
    plt.figure(figsize=(8, 8))
    for x, y, label in object_coords:
        plt.scatter(x, y, label=label)

    # Draw the square
    square_x = [square_start_x, square_end_x, square_end_x, square_start_x, square_start_x]
    square_y = [square_start_y, square_start_y, square_end_y, square_end_y, square_start_y]
    plt.plot(square_x, square_y, 'r-', label='Largest Square')

    # Draw the robot's path
    robot_x = [p[0] for p in robot_path]
    robot_y = [p[1] for p in robot_path]
    plt.plot(robot_x, robot_y, 'b--', label='Robot Path')

    # Mark the center point
    plt.scatter(center_x, center_y, color='black', label='Center Point (Centroid)')

    # Plot aesthetics
    plt.title("Robot Sweeping Path Covering the Square")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True)
    plt.axis('equal')

    # Save the plot as a PNG file
    plt.savefig("robot_sweeping_path.png")
    plt.show()

    # Output dimensions
    print(f"Largest square dimensions: {square_side}m x {square_side}m")
    print(f"Square center point: ({center_x}, {center_y})")
    print(f"Square coordinates: ({square_start_x}, {square_start_y}) to ({square_end_x}, {square_end_y})")

# Test local with object coordinates
make_square([(2.2822766304016113, 4.08355188369751, 'laptop'), 
             (-0.38436222076416016, 0.5081926584243774, 'bottle'), 
             (3.153132915496826, 2.7433860301971436, 'potted plant'), 
             (-0.6404055953025818, -0.3099810779094696, 'suitcase'), 
             (-1.734462022781372, 1.1050945520401, 'umbrella'), 
             (2.3787856101989746, -0.39087170362472534, 'teddy bear'), 
             (2.4763317108154297, 4.186304569244385, 'keyboard'), 
             (3.0441253185272217, -0.12209747731685638, 'stop sign')])
