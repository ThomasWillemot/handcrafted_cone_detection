# handcrafted_cone_detection
Catkin package for cone 3d location estimation

Returns the 3d coordinates of the cones in meters as a ros service.
This service has X,Y and Z components. X is depth(horizontally) , Y is horizontal placement to the left and Z is height.

Run this service using following command: 
```
rosrun handcrafted_cone_detection waypoint_extractor.py
```
call the service using following command:
```
rosservice call /waypoint_extractor_server/rel_cor
```

Python snippet to call the service:
```
#!/usr/bin/env python
import rospy
from handcrafted_cone_detection.srv import SendRelCor

class WaypointClient:
    def __init__(self):
        # Connect to service
        print('Waiting for service')
        rospy.wait_for_service('/waypoint_extractor_server/rel_cor')
        print('service found!')
        self.get_rel_cor = rospy.ServiceProxy('/waypoint_extractor_server/rel_cor', SendRelCor)

    #Retrieves coordinates
        try:
            relative_coordinates = self.get_rel_cor() #calls service and captures response
            print('X,Y,Z coordinates:)
            print(relative_coordinates.x)
            print(relative_coordinates.y)
            print(relative_coordinates.z)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
       

if __name__ == '__main__':
    new_wp_client = WaypointClient()
    new_wp_client.run_test()

```
