# handcrafted_cone_detection
Catkin package for cone 3d location estimation

Returns the 3d coordinates of the cones in meters as a ros service.
This service has X,Y and Z components. X is depth(horizontally) , Y is horizontal placement to the left and Z is height.

Run this service using following command: 
---
rosrun handcrafted_cone_detection waypoint_extractor.py
---
call the service using following command:
---
rosservice call /rel_cor
----
