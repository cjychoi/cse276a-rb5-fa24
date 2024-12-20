# CSE 276A - Intro to Robotics - FA24
Members: Andrew Choi, Nishanth Chidambaram

### Connect to the RB5 robot
``` ssh root@10.42.0.1 ```
``` oelinux123 ```


### Connect to the Internet

```iw phy0 interface add ap0 type __ap```




### Build workspace & Source
  
```source /opt/ros/foxy/setup.bash && cd /root/cse276a_ws && colcon build &&  source install/setup.bash```

### Run camera
  
```ros2 launch rb5_ros2_vision rb_camera_main_ocv_launch.py```


### Source file
```cd src/rb5_ros2/rb5_ros2_vision/launch```


### For ros2 commands on local
- On zsh
  
```micromamba activate ros_env```
```rviz2```


### Navigate to the source code
```cd src/rb5_ros2/rb5_control/src```
``` cd src/rb5_ros2/rb5_ros2_vision/launch ```


### Run on 4 terminals
After sourcing
1. camera

	```ros2 launch rb5_ros2_vision rb_camera_main_ocv_launch.py```
3. twist

	```ros2 run rb5_control mpi_twist_control_node.py```

5. yolo
	
 	```cd src/rb5_ros2/rb5_ros2_vision/launch```

	```python3 yolo_detection_node.py```

7. motion
	
 	```cd src/rb5_ros2/rb5_control/src/ && python3 {solution}```

---
## HW1
[HW1 Report](https://docs.google.com/document/d/1tgoSK-LGrkjnmwbC2iX3vB4yHsQcUyFOQoaKGKj4cTM/edit?tab=t.0)
[Demo Video](https://youtu.be/aajG44xzSN8)

Run hw 1 code
```python3 hw_1.py```

## HW2
[HW2 Report](https://docs.google.com/document/d/1fSxU7LPmJGaLbF6K0kpocTLwebwbZqaJzBPrKvLl9I0/edit?tab=t.0)
[Demo Video](https://youtu.be/z6qyyYL_FeU)

## HW3
[HW3 Report](https://docs.google.com/document/d/12vi6x22ai03davU3_yrkwaHJN8hiRHjM2HlS1EhhmJg/edit?tab=t.0)
[Demo Video](https://youtu.be/4EBOz49WoVc)

Landmark Objects
- ‘laptop’, 'bottle', 'potted plant', 'suitcase', 'umbrella', 'teddy bear', ‘keyboard’, and 'stop sign'

## HW4
[HW4 Report](https://docs.google.com/document/d/1OKB0q6fRi_fIca6uiOqjtX5arRY1b2-387cWHQHw1nY/edit?tab=t.0)

[Demo Video - Maximum Safety](https://youtu.be/O-aZaTxJEKA)

[Demo Video - Minimum Time](https://youtu.be/a3GNTMC-exk)

## HW5
[HW5 Report](https://docs.google.com/document/d/1CBKBt-J4ARTBTceRBNu50OjeHQpOhI9yhpc79_bvuUY/edit?tab=t.0)

[Demo Video](https://youtu.be/Qd-yW8YCU5o)

