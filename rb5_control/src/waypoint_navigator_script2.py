import time
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from mpi_control import MegaPiController

current_position = [0.0, 0.0, 0.0]  # Initialize global array to hold the current position

class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')
        self.get_logger().info("Starting waypoint navigator...")

        # Initialize the MegaPiController
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)  # Allow some time for the connection to establish

        # Control parameters
        self.k_v = 30  # Speed for straight movement
        self.k_w = 55  # Speed for rotational movement
        self.dist_per_sec = 10 / 1  # 10 cm per 1 second at speed 30
        self.rad_per_sec = math.pi / 2  # Pi radians per 2 seconds at speed 55

        # Subscribe to rotation and movement commands from the YoloCameraNode
        self.rotation_sub = self.create_subscription(
            Float32,
            '/rotation_angle',
            self.handle_rotation,
            10
        )
        self.move_sub = self.create_subscription(
            Float32,
            '/move_distance',
            self.handle_movement,
            10
        )

    def handle_rotation(self, msg):
        angle_diff = msg.data
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()
        self.get_logger().info(f"Rotated by {angle_diff} radians.")

    def handle_movement(self, msg):
        distance = msg.data
        movement_time = abs(distance) / (self.dist_per_sec / 100)  # Convert cm/s to m/s
        self.mpi_ctrl.carStraight(self.k_v)
        time.sleep(movement_time)
        self.mpi_ctrl.carStop()
        self.get_logger().info(f"Moved forward by {distance} cm.")

def main(args=None):
    rclpy.init(args=args)
    navigator = WaypointNavigator()
    rclpy.spin(navigator)

    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
