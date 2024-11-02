import time
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from mpi_control import MegaPiController

class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')

        # Initialize the MegaPiController
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)

        # Control parameters
        self.k_v = 30  # Speed for straight movement
        self.k_w = 55  # Speed for rotational movement
        self.dist_per_sec = 10 / 1  # 10 cm per 1 second at speed 30 for straight movement
        self.rad_per_sec = math.pi / 2  # Pi radians per 2 seconds at speed 55 for rotational movement

        # Subscribers
        self.sub_rotate = self.create_subscription(Float32, 'rotate_angle', self.rotate_callback, 10)
        self.sub_move = self.create_subscription(Float32, 'move_distance', self.move_callback, 10)

    def rotate_callback(self, msg):
        angle_diff = msg.data
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

    def move_callback(self, msg):
        distance = msg.data
        movement_time = abs(distance) / (self.dist_per_sec / 100)
        self.mpi_ctrl.carStraight(self.k_v)
        time.sleep(movement_time)
        self.mpi_ctrl.carStop()

def main(args=None):
    rclpy.init(args=args)
    navigator = WaypointNavigator()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
