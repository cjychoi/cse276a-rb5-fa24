class MegaPiControllerNode(Node):
    def __init__(self, verbose=True, debug=False):
        super().__init__("megapi_controller_node")
        self.mpi_ctrl = MegaPiController(port="/dev/ttyUSB0", verbose=verbose)
        self.r = 0.025  # radius of the wheel
        self.lx = 0.055  # half of the distance between front wheel and back wheel
        self.ly = 0.07  # half of the distance between left wheel and right wheel
        self.calibration = 0.25  # Reduce calibration to 0.25 for 4x movement
        self.subscription = self.create_subscription(
            Twist, "/twist", self.twist_callback, 10
        )
        self.subscription

    def twist_callback(self, twist_cmd):
        desired_twist = self.calibration * np.array(
            [[twist_cmd.linear.x], [twist_cmd.linear.y], [twist_cmd.angular.z]]
        )
        print(desired_twist)
        # calculate the jacobian matrix
        jacobian_matrix = (
            np.array(
                [
                    [1, -1, -(self.lx + self.ly)],
                    [1, 1, (self.lx + self.ly)],
                    [1, 1, -(self.lx + self.ly)],
                    [1, -1, (self.lx + self.ly)],
                ]
            )
            / self.r
        )
        # calculate the desired wheel velocity
        result = np.dot(jacobian_matrix, desired_twist)

        # send command to each wheel
        self.mpi_ctrl.setFourMotors(
            int(result[0][0]), int(result[1][0]), int(result[2][0]), int(result[3][0])
        )
