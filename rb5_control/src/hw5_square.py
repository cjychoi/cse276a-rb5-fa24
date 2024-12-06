    # Updated coordinates

import matplotlib.pyplot as plt
import numpy as np

def make_square(object_coords, robot_width=0.17, robot_height=0.20, start_x=0.5, start_y=0.0):
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

    # Find the closest object to the initial center point in each quadrant
    closest_q1 = min(quadrant_1, key=lambda c: c[1] - center_y)
    print('closest q1', closest_q1)
    closest_q2 = min(quadrant_2, key=lambda c: center_x - c[0])
    print('closest q2', closest_q2)
    closest_q3 = min(quadrant_3, key=lambda c: center_y - c[1])
    print('closest q3', closest_q3)
    closest_q4 = min(quadrant_4, key=lambda c: c[0] - center_x)
    print('closest q4', closest_q4)

    square_side = min(closest_q1[1] - closest_q3[1], closest_q4[0] - closest_q2[0]) - margin
    center_x = (closest_q4[0] - closest_q2[0])/2 + closest_q2[0]
    center_y = (closest_q1[1] - closest_q3[1])/2 + closest_q3[1]

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

    (x,y) = closest_q1
    plt.scatter(x, y, color='black', label=label)
    (x,y) = closest_q2
    plt.scatter(x, y, color='black', label=label)
    (x,y) = closest_q3
    plt.scatter(x, y, color='black', label=label)
    (x,y) = closest_q4
    plt.scatter(x, y, color='black', label=label)

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
# make_square([(1.123410701751709, 3.786043405532837, 'laptop'), (-0.18596267700195312, 0.21542099118232727, 'bottle'), (3.3864262104034424, 2.8606085777282715, 'potted plant'), (0.3437591791152954, 3.9257864952087402, 'suitcase'), (1.373099684715271, 0.6129531264305115, 'umbrella'), (-0.8698845505714417, 0.7219505906105042, 'teddy bear'), (1.3916902542114258, 4.341032981872559, 'keyboard'), (3.0175771713256836, -0.2616259753704071, 'stop sign'), (-2.8316409587860107, 1.4042980670928955, 'bicycle'), (-0.3431287407875061, 3.56701397895813, 'bowl'), (0.0, 0.0, 'scissors'), (-1.1141973733901978, 2.5057241916656494, 'backpack')])
