from megapi import MegaPi


MFL = 3  # port for motor front left
MFR = 10  # port for motor front right
MBL = 11  # port for motor back left
MBR = 2  # port for motor back right


class MegaPiController:
    def __init__(self, port="/dev/ttyUSB0", verbose=True):
        self.port = port
        self.verbose = verbose
        if verbose:
            self.printConfiguration()
        self.bot = MegaPi()
        self.bot.start(port=port)
        self.mfl = MFL  # port for motor front left
        self.mfr = MFR  # port for motor front right
        self.mbl = MBL  # port for motor back left
        self.mbr = MBR  # port for motor back right

    def printConfiguration(self):
        print("MegaPiController:")
        print("Communication Port:" + repr(self.port))
        print(
            "Motor ports: "
            + " MFL: "
            + repr(MFL)
            + " MFR: "
            + repr(MFR)
            + " MBL: "
            + repr(MBL)
            + " MBR: "
            + repr(MBR)
        )

    def setFourMotors(self, vfl=0, vfr=0, vbl=0, vbr=0):
        if self.verbose:
            print(
                "Set Motors: vfl: "
                + repr(int(round(vfl, 0)))
                + " vfr: "
                + repr(int(round(vfr, 0)))
                + " vbl: "
                + repr(int(round(vbl, 0)))
                + " vbr: "
                + repr(int(round(vbr, 0)))
            )
        self.bot.motorRun(self.mfl, -vfl)
        self.bot.motorRun(self.mfr, vfr)
        self.bot.motorRun(self.mbl, -vbl)
        self.bot.motorRun(self.mbr, vbr)

    def carStop(self):
        if self.verbose:
            print("\nCAR STOP:")
        self.setFourMotors()

    def carStraight(self, speed):
        if self.verbose:
            print("\nCAR STRAIGHT:")
        self.setFourMotors(speed, speed, speed, speed)

    def carRotate(self, speed):
        if self.verbose:
            print("\nCAR ROTATE:")
        self.setFourMotors(-speed, speed, -speed, speed)

    def carSlide(self, speed):
        if self.verbose:
            print("\nCAR SLIDE:")
        self.setFourMotors(-speed , speed*2, speed*2, -speed)

    def carFowardLeft(self, speed):
        if self.verbose:
            print("\nCAR FORWARD LEFT:")
        self.setFourMotors(0 , speed*2, speed*2, 0)

    def carMixed(self, v_straight, v_rotate, v_slide):
        if self.verbose:
            print("CAR MIXED")
        self.setFourMotors(
            v_rotate - v_straight + v_slide,
            v_rotate + v_straight + v_slide,
            v_rotate - v_straight - v_slide,
            v_rotate + v_straight - v_slide,
        )

    def close(self):
        self.bot.close()
        self.bot.exit()


if __name__ == "__main__":
    import time

    mpi_ctrl = MegaPiController(port="/dev/ttyUSB0", verbose=True)
    time.sleep(1)

    # FAH 3F Carpet Calibration

        # Slide 50 - 1s = 0.1m
        # mpi_ctrl.carSlide(50)
        # time.sleep(1)

        # Straight 60 - 0.7s = 0.1m
        # mpi_ctrl.carStraight(50)
        # time.sleep(0.5)

        # Rotate 55 - 1s = 90 deg
        # mpi_ctrl.carRotate(60)
        # time.sleep(1)

    mpi_ctrl.carStraight(50)
    time.sleep(1)

    mpi_ctrl.carRotate(55)
    time.sleep(2)

    mpi_ctrl.carStraight(50)
    time.sleep(1)

    mpi_ctrl.carRotate(55)
    time.sleep(2)

    # mpi_ctrl.carSlide(60)
    # time.sleep(0.7)

    # mpi_ctrl.carRotate(60)
    # time.sleep(1)

    # mpi_ctrl.carStraight(50)
    # time.sleep(0.5)

    # mpi_ctrl.carStraight(30)
    # time.sleep(1)
    # mpi_ctrl.carRotate(55)
    # time.sleep(1)

    # mpi_ctrl.carStraight(50)
    # time.sleep(4)
    # mpi_ctrl.carRotate(-60)
    # time.sleep(1)
    # mpi_ctrl.carStraight(30)
    # time.sleep(1)
    # mpi_ctrl.carRotate(-60)
    # time.sleep(1)

    # mpi_ctrl.carStraight(50)
    # time.sleep(4)
    # mpi_ctrl.carRotate(60)
    # time.sleep(1)
    # mpi_ctrl.carStraight(30)
    # time.sleep(1)
    # mpi_ctrl.carRotate(60)
    # time.sleep(1)

    # mpi_ctrl.carStraight(50)
    # time.sleep(4)
    # mpi_ctrl.carRotate(-60)
    # time.sleep(1)
    # mpi_ctrl.carStraight(30)
    # time.sleep(1)
    # mpi_ctrl.carRotate(-60)
    # time.sleep(1)

    # mpi_ctrl.carStraight(-50)
    # time.sleep(4)
    # mpi_ctrl.carFowardLeft(70)
    # time.sleep(2)
    # mpi_ctrl.carStraight(50)
    # time.sleep(4)
    # mpi_ctrl.carFowardLeft(70)
    # time.sleep(2)
    # mpi_ctrl.carStraight(-50)
    # time.sleep(4)

    mpi_ctrl.carStop()
    # print("If your program cannot be closed properly, check updated instructions in google doc.")
