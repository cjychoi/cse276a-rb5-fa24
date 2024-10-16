#!/usr/bin/env python3

"""
Copyright 2023, UC San Diego, Contextual Robotics Institute

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import rclpy
from rclpy.node import Node
from mpi_control import MegaPiController
from std_msgs.msg import String
from sensor_msgs.msg import Joy


class KeyJoy_MegaPiController(Node):
    def __init__(self, verbose=True, debug=False):
        super().__init__('key_joy_controller_node')
        self.bot = MegaPiController(port='/dev/ttyUSB0', verbose=verbose)
        self.verbose = verbose
        if self.verbose:
            self.logger_ = self.get_logger()
        self.debug = debug
        self.subscription = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.subscription  # prevent unused variable warning


    def move(self, direction, speed):
        if direction == "left":
            if self.verbose:
                self.logger_.info("Moving left")
            self.bot.setFourMotors(speed, speed, -speed, -speed)

        elif direction == "right":
            if self.verbose:
                self.logger_.info("Moving right")
            self.bot.setFourMotors(-speed, -speed, speed, speed)

        elif direction == "forward":
            if self.verbose:
                self.logger_.info("Moving forward")
            self.bot.setFourMotors(-speed, speed, -speed, speed)

        elif direction == "reverse":
            if self.verbose:
                self.logger_.info("Moving in reverse")
            self.bot.setFourMotors(speed, -speed, speed, -speed)

        elif direction == "ccwise":
            # TODO: Counter clockwise
            if self.verbose:
                self.logger_.info("Moving counter clockwise")
            self.bot.setFourMotors(speed, speed, -speed, -speed)

        elif direction == "cwise":
            # TODO: clockwise
            if self.verbose:
                self.logger_.info("Moving clockwise")
            self.bot.setFourMotors(speed, speed, -speed, -speed)

        else:
            if self.verbose:
                self.logger_.info("Stopping")
            self.bot.setFourMotors(0, 0, 0, 0)


    def joy_callback(self, msg):
        
        if msg.axes[0] > 0.0:
            # left
            self.move("left", 30)
        elif msg.axes[0] < 0.0:
            # right
            self.move("right", 30)
        elif msg.axes[1] > 0.0:
            # forward
            self.move("forward", 30)
        elif msg.axes[1] < 0.0:
            # reverse
            self.move("reverse", 30)
        elif msg.axes[2] < 0.0:
            # turn clock-wise 
            self.move("cwise", 30)
        elif msg.axes[2] > 0.0:
            # turn counter clock-wise 
            self.move("ccwise", 30)
        else:
            self.move("stop", 0)




if __name__ == "__main__":
    rclpy.init()
    key_joy_controller = KeyJoy_MegaPiController()
    rclpy.spin(key_joy_controller)

    key_joy_controller.destroy_node()
    rclpy.shutdown()
