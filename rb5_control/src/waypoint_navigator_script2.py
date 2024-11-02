import time
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from mpi_control import MegaPiController

current_position = [0.0, 0.0, 0.0]

class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')

        # Initialize the MegaPiController
        self.mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)

        # Control parameters
        self.k_v = 30
        self.k_w = 55
        self.dist_per_sec = 10 / 1
        self.rad_per_sec = math.pi / 2
        self.tolerance = 0.1

        # Subscribe to rotation and movement commands
        self.rotation_subscriber = self.create_subscription(Float32, '/rotate_angle', self.handle_rotation, 10)
        self.distance_subscriber = self.create_subscription(Float32, '/move_distance', self.handle_move_straight, 10)

        # Publisher to notify rotation complete
        self.rotation_complete_publisher = self.create_publisher(Float32, '/rotation_complete', 10)

    def handle_rotation(self, msg):
        angle_diff = msg.data
        self.rotate_to_angle(angle_diff)
        # Notify that rotation is complete
        complete_msg = Float32()
        complete_msg.data = angle_diff
        self.rotation_complete_publisher.publish(complete_msg)

    def handle_move_straight(self, msg):
        distance = msg.data
        self.move_straight(distance)

    def rotate_to_angle(self, angle_diff):
        rotation_time = abs(angle_diff) / self.rad_per_sec
        self.mpi_ctrl.carRotate(self.k_w if angle_diff > 0 else -self.k_w)
        time.sleep(rotation_time)
        self.mpi_ctrl.carStop()

    def move_straight(self, distance):
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
