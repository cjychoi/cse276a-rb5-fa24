# First steps in ubuntu

ssh root@10.42.0.1

oelinux123

cd /root/cse276a_ws/src/rb5_ros2 && source /opt/ros/foxy/setup.bash && cd /root/cse276a_ws && colcon build && source install/setup.bash && cd src/rb5_ros2/rb5_ros2_control/rb5_ros2_control

# to run hw 1 code
python3 hw_1.py

# Rb5 related packages in ROS2

These packages are developed using ROS Foxy on Ubuntu 20.04.

## 1. rb5_ros2_vision

use the following command to launch the main camera on rb5

``` bash
ros2 launch rb5_ros2_vision rb_camera_main_ocv_launch.py
```
make sure that 'use_rb_cam' parameter is set to true.

use the following command to launch on a webcam
``` bash
ros2 launch rb5_ros2_vision rb_camera_webcam_ocv_launch.py
```
make sure that 'use_rb_cam' parameter is set to false.

If 'image_rectify' is set to true, make sure you set the 'camera_parameter_path'
correctly to a xml or yaml/yml file. (tested on yaml) Notice that newer versions of opencv saves matrices with 'dt' representing data type. Also the file must start with the yaml version.

## Notes on other versions of ROS2

1. In launch files, dashing uses node_executable while foxy uses executable.
2. Dashing doesn't support command line arguements. 

