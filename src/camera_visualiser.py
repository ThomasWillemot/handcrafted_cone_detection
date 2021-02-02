# !/usr/bin/python3.8

import numpy as np
import rospy
from sensor_msgs.msg import Image
from handcrafted_cone_detection.msg import ConeImgLoc


class CameraVisualiser:

    def __init__(self):
        rospy.Subscriber("/cone_coordin", ConeImgLoc, self._publish_combined_global_poses)
        self._publisher = rospy.Publisher('/annotated_cone',
                                          Image, queue_size=10)
        rospy.init_node('camera_visualiser')

    def _publish_combined_global_poses(self, data) -> None:
        resolution = (800, 848)
        position, width, height = (data.x_pos, data.y_pos), data.cone_width, 5
        frame = np.zeros(resolution)
        frame[position[0]:position[0] + width,
        position[1]:position[1] + height] = 255

        image = Image()
        image.data = frame.astype(np.uint8).flatten().tolist()
        image.height = resolution[0]
        image.width = resolution[1]
        image.encoding = 'mono8'
        image.step = resolution[1]
        self._publisher.publish(image)

    def _process_state_and_publish_frame(self, data: Image):
        self._publish_combined_global_poses(data)

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == "__main__":
    publisher = CameraVisualiser()
    publisher.run()
