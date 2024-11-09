# slam_control_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import matplotlib.pyplot as plt

class EKFSLAM:
    def __init__(self, object_list):
        # Initialize SLAM state (robot pose and landmarks)
        self.state = np.zeros((3 + 2 * len(object_list), 1))

    def update(self, measurement, obj_index):
        # Placeholder for EKF update
        pass

    def get_state(self):
        return self.state

class SlamControlNode(Node):
    def __init__(self):
        super().__init__('slam_control_node')

        # Publisher for robot motion (Twist)
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)

        # Subscriber for object detection info
        self.subscription = self.create_subscription(
            Float32MultiArray, '/detected_object_info', self.object_callback, 10
        )

        # Robot state variables
        self.robot_position = np.array([0.0, 0.0])  # Initial position (x, y)
        self.robot_orientation = 0.0  # Initial orientation (theta in radians)
        self.velocity = np.array([0.0, 0.0])  # Velocity (linear.x, angular.z)

        # Movement control
        self.is_moving = False
        self.is_rotating = False
        self.move_duration = 0.0
        self.turn_duration = 0.0
        self.movement_timer = None

        # Initialize EKF SLAM
        self.objects_to_detect = ['teddy bear', 'backpack', 'umbrella', 'bottle']
        self.ekf_slam = EKFSLAM(self.objects_to_detect)

        # Initialize plot for real-time visualization
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-5, 5)  # Adjust limits based on environment size
        self.ax.set_ylim(-5, 5)
        self.robot_line, = self.ax.plot([], [], 'bo', label="Robot")
        self.robot_positions = []  # Store positions for plotting

        # Subscriber for Twist messages from MegaPi controller
        self.twist_subscription = self.create_subscription(
            Twist, '/twist', self.twist_callback, 10
        )

        # Update the robot's position every 0.5 seconds
        self.timer = self.create_timer(0.5, self.update_position)

        # Start the movement after subscribing
        self.spin_and_track()

    def object_callback(self, msg):
        print("\n\n<<Object Callback>>\n\n")
        distance, angle = msg.data
        obj_index = 0  # Placeholder for the detected object's index
        self.ekf_slam.update(np.array([distance, angle]), obj_index)

    def twist_callback(self, msg):
        # Store the current velocities from the Twist message
        self.velocity = np.array([msg.linear.x, msg.angular.z])

    def update_position(self):
        if not self.is_moving and not self.is_rotating:
            return

        # Integrate to find the robot's new position
        delta_x = self.velocity[0] * 0.5 * np.cos(self.robot_orientation)
        delta_y = self.velocity[0] * 0.5 * np.sin(self.robot_orientation)
        self.robot_position += np.array([delta_x, delta_y])
    
        # Update robot's orientation
        self.robot_orientation += self.velocity[1] * 0.5
    
        # Save current position to array
        self.robot_positions.append(self.robot_position.copy())

    def spin_and_track(self):
        # Move the robot in a 2m x 2m square
        for i in range(4):  # For a square
            self.move_forward(2.0)  # Move 2.0 meters
            self.turn_90_degrees()  # Turn 90 degrees
    
        # After completing the movement, plot all saved positions
        self.plot_robot_positions()
    
        # Save the plot after plotting
        self.save_plot()

    def move_forward(self, distance):
        # Move the robot forward by a specified distance (in meters)
        print("\n\n<<Move Forward>>\n\n")
        move_twist = Twist()
        move_twist.linear.x = 2.0  # Set forward speed

        # Calculate the time to move the distance based on speed
        self.move_duration = distance / 2.0  # Time needed to move 2m at 2m/s
        self.is_moving = True

        # Publish the movement command
        self.publisher_.publish(move_twist)

        # Start a timer to stop after moving
        self.movement_timer = self.create_timer(self.move_duration, self.stop_moving)

    def stop_moving(self):
        # Stop the robot after moving the specified distance
        move_twist = Twist()
        move_twist.linear.x = 0.0  # Stop the robot
        self.publisher_.publish(move_twist)

        self.is_moving = False
        self.movement_timer.cancel()

    def turn_90_degrees(self):
        print("\n\n<<Turn 90 degrees>>\n\n")
        turn_twist = Twist()
        turn_twist.angular.z = 9.0  # Set angular speed to rotate 90 degrees

        # Time to rotate 90 degrees
        self.turn_duration = 1.57 / 9.0  # 1.57 radians / speed = time
        self.is_rotating = True

        # Publish the rotation command
        self.publisher_.publish(turn_twist)

        # Start a timer to stop rotating after 90 degrees
        self.movement_timer = self.create_timer(self.turn_duration, self.stop_rotating)

    def stop_rotating(self):
        # Stop the robot after rotating
        turn_twist = Twist()
        turn_twist.angular.z = 0.0  # Stop rotating
        self.publisher_.publish(turn_twist)

        self.is_rotating = False
        self.movement_timer.cancel()

    def plot_robot_positions(self):
        # Convert saved positions to a numpy array
        if len(self.robot_positions) > 0:
            robot_positions_array = np.array(self.robot_positions)
            
            # Check that robot_positions_array has the correct shape
            if robot_positions_array.ndim == 2 and robot_positions_array.shape[1] == 2:
                # Plot each saved position as a point
                self.ax.plot(robot_positions_array[:, 0], robot_positions_array[:, 1], 'bo')
        
        # Re-draw the plot
        self.ax.legend(loc='upper right')
        plt.show()

    def save_plot(self):
        print("\n\n<<Saving Plot>>\n\n")
        self.fig.savefig('slam_plot.png')  # Save the plot as an image file

def main(args=None):
    rclpy.init(args=args)
    slam_node = SlamControlNode()

    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass

    slam_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
