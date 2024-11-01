import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np

class PIDcontroller(Node):
    def __init__(self, Kp, Ki, Kd):
        super().__init__('pid_controller_node')
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = None
        self.I = np.array([0.0, 0.0, 0.0])
        self.lastError = np.array([0.0, 0.0, 0.0])
        self.timestep = 0.1
        self.maximumValue = 0.1
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)

    def setTarget(self, state):
        self.I = np.array([0.0, 0.0, 0.0])
        self.lastError = np.array([0.0, 0.0, 0.0])
        self.target = np.array(state)

    def getError(self, currentState, targetState):
        result = targetState - currentState
        result[2] = (result[2] + np.pi) % (2 * np.pi) - np.pi
        return result 

    def update(self, currentState):
        e = self.getError(currentState, self.target)
        P = self.Kp * e
        self.I = self.I + self.Ki * e * self.timestep 
        D = self.Kd * (e - self.lastError)
        result = P + self.I + D

        if np.linalg.norm(result) > self.maximumValue:
            result = (result / np.linalg.norm(result)) * self.maximumValue
            self.I = 0.0

        self.lastError = e
        self.publisher_.publish(self.genTwistMsg(result))
        self.log_position(currentState + result)

    def genTwistMsg(self, desired_twist):
        twist_msg = Twist()
        twist_msg.linear.x = desired_twist[0]
        twist_msg.linear.y = desired_twist[1]
        twist_msg.angular.z = desired_twist[2]
        return twist_msg

    def log_position(self, position):
        with open('/path/to/position_log.csv', 'a') as f:
            f.write(f"{position[0]},{position[1]},{position[2]}\n")
