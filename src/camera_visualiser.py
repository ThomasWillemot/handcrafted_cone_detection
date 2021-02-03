# !/usr/bin/python3.8

import numpy as np
import rospy
from sensor_msgs.msg import Image
from handcrafted_cone_detection.msg import ConeImgLoc
from cv_bridge import CvBridge
import cv2
import time
class CameraVisualiser:

    def __init__(self):
        self.x_position = 0
        self.y_position = 0
        self.cone_width = 1
        dim = (848, 800)
        k = np.array(
            [[285.95001220703125, 0.0, 418.948486328125], [0.0, 286.0592956542969, 405.756103515625], [0.0, 0.0, 1.0]])
        d = np.array(
            [[-0.006003059912472963], [0.04132957011461258], [-0.038822319358587265], [0.006561396177858114]])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, dim,
                                                                   cv2.CV_16SC2)
        self.bridge = CvBridge()
        rospy.Subscriber("/cone_coordin", ConeImgLoc, self.save_last_2d_coor)
        rospy.Subscriber("/camera/fisheye1/image_raw", Image, self._publish_combined_global_poses)
        self._publisher = rospy.Publisher('/annotated_cone',
                                          Image, queue_size=10)

        rospy.init_node('camera_visualiser')

    def save_last_2d_coor(self, data):
        self.x_position = data.x_pos
        self.y_position = data.y_pos
        self.cone_width = data.cone_width

    def _publish_combined_global_poses(self, data: Image) -> None:
        cv_im = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')  # Load images to cv
        rect_image = cv2.remap(cv_im, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)  # Remap fisheye to normal picture
        resolution = (800, 848)
        frame = np.array(rect_image)
        frame = cv2.circle(frame, (self.x_position, self.y_position), int(self.cone_width/2), 255,5)
        image = Image()
        image.data = frame.astype(np.uint8).flatten().tolist()
        image.height = resolution[0]
        image.width = resolution[1]
        image.encoding = 'mono8'
        image.step = resolution[1]
        self._publisher.publish(image)
    def run(self):
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == "__main__":
    publisher = CameraVisualiser()
    publisher.run()
