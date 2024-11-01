from mpi_control import MegaPiController
import time

def main():
    mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
    time.sleep(1)

#   1 second at speed 30 is 10 cm forward
    mpi_ctrl.carStraight(30)  # Move straight with speed 30
    time.sleep(1*5)  # default 1 for 10 cm

#   1 seconds at speed 55 is 45 degrees clockwise 
#    mpi_ctrl.carRotate(55)  # Rotate with speed 55
#    time.sleep(1*2)  # default 1 for 45 degrees

#    mpi_ctrl.carStraight(30)
#    time.sleep(1*7)
#    mpi_ctrl.carRotate(56)
#    time.sleep(1*2)
#    mpi_ctrl.carStraight(30)
#    time.sleep(1*2)

    mpi_ctrl.carStop()  # Stop the car
    mpi_ctrl.close()  # Close the controller

if __name__ == "__main__":
    main()
#