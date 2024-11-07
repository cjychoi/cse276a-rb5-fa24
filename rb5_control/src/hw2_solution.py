#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, PoseArray
import numpy as np
import math
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from rclpy.time import Duration
import math
from math import copysign, fabs, sqrt, pi, sin, cos, asin, acos, atan2, exp, log


EPSILON = 1e-0  # threshold for computing quaternion from rotation matrix


def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.empty((4,))
    q[0] = cj * sc - sj * cs
    q[1] = cj * ss + sj * cc
    q[2] = cj * cs - sj * sc
    q[3] = cj * cc + sj * ss

    return q


def quaternion_to_rotation_matrix(q):
    """
    Convert a unit quaternion to a rotation matrix.

    This function is only for internal usage. Please use the following
    instead when it is needed to do this conversion.

        R = q.to_rotation_matrix()

    """
    x, y, z, w = q
    w2 = w * w
    x2 = x * x
    y2 = y * y
    z2 = z * z

    # Check the norm of the quaternion! It must be a unit quaternion!
    assert fabs(w2 + x2 + y2 + z2 - 1) < 1e-6

    wx = 2 * w * x
    wy = 2 * w * y
    wz = 2 * w * z

    xy = 2 * x * y
    xz = 2 * x * z
    yz = 2 * y * z

    R = np.asarray(
        (
            (w2 + x2 - y2 - z2, xy - wz, xz + wy),
            (xy + wz, w2 - x2 + y2 - z2, yz - wx),
            (xz - wy, yz + wx, w2 - x2 - y2 + z2),
        )
    )

    return R


def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a unit quaternion.

    This uses the Shepperd’s method for numerical stabilty.

    This function is only for internal usage. Please use the following
    instead when it is needed to do this conversion.

        q = Quaternion.construct_from_rotation_matrix(R)

    """

    # The rotation matrix must be orthonormal
    assert np.allclose(np.dot(R, R.conj().transpose()), np.eye(3), atol=EPSILON)

    # Check the determinant of R! It must be 1.
    assert math.isclose(np.linalg.det(R), 1, abs_tol=EPSILON)

    w2 = 1 + R[0, 0] + R[1, 1] + R[2, 2]
    x2 = 1 + R[0, 0] - R[1, 1] - R[2, 2]
    y2 = 1 - R[0, 0] + R[1, 1] - R[2, 2]
    z2 = 1 - R[0, 0] - R[1, 1] + R[2, 2]

    yz = R[1, 2] + R[2, 1]
    xz = R[2, 0] + R[0, 2]
    xy = R[0, 1] + R[1, 0]

    wx = R[2, 1] - R[1, 2]
    wy = R[0, 2] - R[2, 0]
    wz = R[1, 0] - R[0, 1]

    if R[2, 2] < 0:

        if R[0, 0] > R[1, 1]:

            x = sqrt(x2)
            w = wx / x
            y = xy / x
            z = xz / x

        else:

            y = sqrt(y2)
            w = wy / y
            x = xy / y
            z = yz / y

    else:

        if R[0, 0] < -R[1, 1]:

            z = sqrt(z2)
            w = wz / z
            x = xz / z
            y = yz / z

        else:

            w = sqrt(w2)
            x = wx / w
            y = wy / w
            z = wz / w

    w = w * 0.5
    x = x * 0.5
    y = y * 0.5
    z = z * 0.5

    return x, y, z, w


def combine_transformations(R1, t1, R2, t2):
    """
    Combine two transformations given by rotation matrices and translation vectors.

    Parameters:
    R1 (numpy.ndarray): First 3x3 rotation matrix.
    t1 (numpy.ndarray): First translation vector, shape (3,).
    R2 (numpy.ndarray): Second 3x3 rotation matrix.
    t2 (numpy.ndarray): Second translation vector, shape (3,).

    Returns:
    R_combined (numpy.ndarray): Combined 3x3 rotation matrix.
    t_combined (numpy.ndarray): Combined translation vector, shape (3,).
    """
    # Ensure the input matrices and vectors have correct shapes
    assert R1.shape == (3, 3), "R1 must be a 3x3 matrix"
    assert t1.shape == (3,), "t1 must be a 3-element vector"
    assert R2.shape == (3, 3), "R2 must be a 3x3 matrix"
    assert t2.shape == (3,), "t2 must be a 3-element vector"

    T1 = np.eye(4)
    T1[:3, :3] = R1
    T1[:3, 3] = t1

    T2 = np.eye(4)
    T2[:3, :3] = R2
    T2[:3, 3] = t2

    return np.dot(T1, T2)


class PIDcontroller(Node):
    def __init__(self, Kp, Ki, Kd):
        super().__init__("PID_Controller_NodePub")
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = None
        self.I = np.array([0.0, 0.0, 0.0])
        self.lastError = np.array([0.0, 0.0, 0.0])
        self.timestep = 0.1
        self.maximumValue = 0.02
        self.publisher_ = self.create_publisher(Twist, "/twist", 10)

    def setTarget(self, target):
        """
        Set the target pose.
        """
        self.I = np.array([0.0, 0.0, 0.0])
        self.lastError = np.array([0.0, 0.0, 0.0])
        self.target = np.array(target)

    def getError(self, currentState, targetState):
        """
        Return the difference between two states.
        """
        result = targetState - currentState
        result[2] = (result[2] + np.pi) % (2 * np.pi) - np.pi
        return result

    def setMaximumUpdate(self, mv):
        """
        Set maximum velocity for stability.
        """
        self.maximumValue = mv

    def update(self, currentState):
        """
        Calculate the update value on the state based on the error between current state and target state with PID.
        """
        e = self.getError(currentState, self.target)
        P = self.Kp * e
        self.I += self.Ki * e * self.timestep
        D = self.Kd * (e - self.lastError)
        result = P + self.I + D
        self.lastError = e

        # Scale down the twist if its norm is more than the maximum value
        resultNorm = np.linalg.norm(result)
        if resultNorm > self.maximumValue:
            result = (result / resultNorm) * self.maximumValue
            self.I = 0.0

        return result


class RobotStateEstimator(Node):
    def __init__(self):
        super().__init__("robot_state_estimator")
        self.subscription = self.create_subscription(
            PoseArray, "/april_poses", self.april_pose_callback, 10
        )
        self.subscription
        self.br = TransformBroadcaster(self)
        self.current_state = np.array([0.0, 0.0, 0.0])
        self.pose_updated = True

        # if you are not using tf2 service, define your aptiltag poses (wrt map frame) here, z value is omitted to ensure 2D transformation
        self.apriltag_world_poses = {
            "marker_0": (
                1.34,
                0.0,
                0.0,
                0.5,
                -0.5,
                0.5,
                -0.5,
            ),  # (x,y,z, qual_x, qual_y, qual_z, qual_w)
            "marker_1": (-0.5, 2.0, 0.0, -0.5, -0.5, 0.5, 0.5),
        }

    def april_pose_callback(self, msg):
        # Log the number of poses in the PoseArray
        self.pose_updated = False
        self.get_logger().info(f"Received PoseArray with {len(msg.poses)} poses")
        if len(msg.poses) < 1:
            return
        pose_ids = msg.header.frame_id.split(",")[:-1]

        # we will only use one landmark at a time in homework 2. in homework 3, all landmarks should be considered.
        tag_id = pose_ids[0]
        pose_camera_apriltag = msg.poses[0]  # syntax: pose_ReferenceFrame_TargetFrame
        trans_camera_apriltag = np.array(
            [
                pose_camera_apriltag.position.x,
                pose_camera_apriltag.position.y,
                pose_camera_apriltag.position.z,
            ]
        )
        quat_camera_apriltag = np.array(
            [
                pose_camera_apriltag.orientation.x,
                pose_camera_apriltag.orientation.y,
                pose_camera_apriltag.orientation.z,
                pose_camera_apriltag.orientation.w,
            ]
        )
        rot_camera_apriltag = quaternion_to_rotation_matrix(quat_camera_apriltag)

        # project to 2d
        trans_camera_apriltag_2d = trans_camera_apriltag
        trans_camera_apriltag_2d[1] = 0.0

        rot_camera_apriltag_2d = np.array(
            [
                [rot_camera_apriltag[0, 0], 0.0, rot_camera_apriltag[2, 0]],
                [0.0, 1.0, 0.0],
                [rot_camera_apriltag[2, 0], 0.0, rot_camera_apriltag[0, 0]],
            ]
        )

        rot_apriltag_camera_2d = rot_camera_apriltag_2d.T
        trans_apriltag_camera_2d = -np.dot(
            rot_apriltag_camera_2d, trans_camera_apriltag_2d
        )

        # retrieve static transformation of apriltags wrt map
        rot_map_apriltag = quaternion_to_rotation_matrix(
            self.apriltag_world_poses[tag_id][3:]
        )
        trans_map_apriltag = np.asarray(self.apriltag_world_poses[tag_id][:3])

        # apply transfomation
        T_map_camera = combine_transformations(
            rot_map_apriltag,
            trans_map_apriltag,
            rot_apriltag_camera_2d,
            trans_apriltag_camera_2d,
        )
        rot_map_camera, trans_map_camera = T_map_camera[:3, :3], T_map_camera[:3, 3]
        angle = math.atan2(rot_map_camera[1][2], rot_map_camera[0][2])

        self.current_state = np.array([trans_map_camera[0], trans_map_camera[1], angle])
        self.pose_updated = True


def genTwistMsg(desired_twist):
    """
    Convert the twist to twist msg.
    """
    twist_msg = Twist()
    twist_msg.linear.x = desired_twist[0]
    twist_msg.linear.y = desired_twist[1]
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = desired_twist[2]
    return twist_msg


def coord(twist, current_state):
    J = np.array(
        [
            [np.cos(current_state[2]), np.sin(current_state[2]), 0.0],
            [-np.sin(current_state[2]), np.cos(current_state[2]), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return np.dot(J, twist)


def main(args=None):
    rclpy.init(args=args)
    robot_state_estimator = RobotStateEstimator()
    waypoint = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, np.pi], [0.0, 0.0, 0.0]]
    )

    # init pid controller
    pid = PIDcontroller(0.1, 0.005, 0.005)
    current_state = robot_state_estimator.current_state

    for wp in waypoint:
        print("move to way point", wp)
        # set wp as the target point
        pid.setTarget(wp)

        # calculate the current twist
        update_value = pid.update(current_state)
        # publish the twist
        pid.publisher_.publish(genTwistMsg(coord(update_value, current_state)))
        # print(coord(update_value, current_state))
        time.sleep(0.05)
        # update the current state
        current_state += update_value
        rclpy.spin_once(robot_state_estimator)
        found_state, estimated_state = (
            robot_state_estimator.pose_updated,
            robot_state_estimator.current_state,
        )
        if (
            found_state
        ):  # if the tag is detected, we can use it to update current state.
            current_state = estimated_state
        while (
            np.linalg.norm(pid.getError(current_state, wp)) > 0.05
        ):  # check the error between current state and current way point
            # calculate the current twist
            update_value = pid.update(current_state)
            # publish the twist
            pid.publisher_.publish(genTwistMsg(coord(update_value, current_state)))
            # print(coord(update_value, current_state))
            time.sleep(0.05)
            # update the current state
            current_state += update_value
            rclpy.spin_once(robot_state_estimator)
            found_state, estimated_state = (
                robot_state_estimator.pose_updated,
                robot_state_estimator.current_state,
            )
            if (
                found_state
            ):  # if the tag is detected, we can use it to update current state.
                print('\n\n\n\n---------------tag found--------------\n\n\n\n')
                current_state = estimated_state
    # stop the car and exit
    pid.publisher_.publish(genTwistMsg(np.array([0.0, 0.0, 0.0])))

    robot_controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
