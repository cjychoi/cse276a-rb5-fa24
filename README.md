# CSE 276A - Intro to Robotics - FA24
Members: Andrew Choi (A69033628), Nishanth Chidambaram (A69031827)

### Connect to the RB5 robot
``` ssh root@10.42.0.1 ```
``` oelinux123 ```

### Navigate to the source code
```cd src/rb5_ros2/rb5_ros2_control/rb5_ros2_control```

```cd /root/cse276a_ws/src/rb5_ros2 && source /opt/ros/foxy/setup.bash && cd /root/cse276a_ws && colcon build && source install/setup.bash && cd src/rb5_ros2/rb5_ros2_control/rb5_ros2_control```


- Connect to Internet

```iw phy0 interface add ap0 type __ap```
```cd /root/cse276a_ws/src/rb5_ros2```


- Build workspace & Source
  
```source /opt/ros/foxy/setup.bash```
```cd /root/cse276a_ws && colcon build```
```source install/setup.bash```


- Run camera
  
```ros2 launch rb5_ros2_vision rb_camera_main_ocv_launch.py```


- For ros2 commands on local
- On zsh
  
```micromamba activate ros_env```
```rviz2```

---
## HW1
[HW1 Report](https://docs.google.com/document/d/1tgoSK-LGrkjnmwbC2iX3vB4yHsQcUyFOQoaKGKj4cTM/edit?tab=t.0)
[Demo Video](https://youtu.be/aajG44xzSN8)

Run hw 1 code
```python3 hw_1.py```
