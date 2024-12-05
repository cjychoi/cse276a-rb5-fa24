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

    # NEW CODE: Find the closest object in each quadrant and add to new_centroid array
    def closest_point(quadrant):
        return min(quadrant, key=lambda p: np.hypot(p[0] - center_x, p[1] - center_y)) if quadrant else None

    new_centroid = []
    for quadrant in [quadrant_1, quadrant_2, quadrant_3, quadrant_4]:
        closest = closest_point(quadrant)
        if closest:
            new_centroid.append(closest)

    # Extract x and y coordinates
    # x_coords = np.array([coord[0] for coord in object_coords])
    # y_coords = np.array([coord[1] for coord in object_coords])

    # # Calculate the center point (centroid)
    # center_x = np.mean(x_coords)
    # center_y = np.mean(y_coords)

    # Print new_centroid array
    print("New centroid array (closest objects in each quadrant):", new_centroid)

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

    # for x,y in new_centroid:
    #     plt.scatter(x,y,color='black',label='new centroid')

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
make_square([(1.4415265321731567, 4.385348796844482, 'laptop'), (0.01592075452208519, 0.08464153110980988, 'bottle'), (3.2468042373657227, 2.861740827560425, 'potted plant'), (0.15551592409610748, 3.771833658218384, 'suitcase'), (1.6348110437393188, 0.00795849971473217, 'umbrella'), (1.540797472000122, -0.27932679653167725, 'teddy bear'), (1.4621721506118774, 4.550741672515869, 'keyboard'), (3.6618740558624268, 0.19870199263095856, 'stop sign'), (-1.935784935951233, 1.086277961730957, 'bicycle'), (-0.8318294882774353, 3.6188526153564453, 'bowl'), (0.0, 0.0, 'scissors'), (-0.4886416792869568, 2.4206247329711914, 'backpack')])