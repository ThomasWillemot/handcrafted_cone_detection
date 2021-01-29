# !/usr/bin/python3.8
"""
Modifies sensor information to a state or observation used by rosenvironment.
Defines separate modes:
Default mode:
    combine tf topics of tracking and fleeing agent in global frame
"""
import time

import numpy as np
import rospy
from sensor_msgs.msg import Image

from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension
from src.sim.ros.src.utils import process_odometry, process_pose_stamped, euler_from_quaternion, get_output_path, \
    process_combined_global_poses, array_to_combined_global_pose, calculate_bounding_box


class CameraVisualiser:

    def __init__(self):
        stime = time.time()
        max_duration = 60
        while not rospy.has_param('/waypoint_extractor_server') and time.time() < stime + max_duration:
            time.sleep(0.01)

        self._output_path = get_output_path()
        self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)

        rospy.Subscriber(rospy.get_param('/robot/modified_state_sensor/topic', '/modified_state'),
                         eval(rospy.get_param('/robot/modified_state_sensor/type', 'CombinedGlobalPoses')),
                         self._process_state_and_publish_frame)
        self._publisher = rospy.Publisher('/modified_state_frame',
                                          Image, queue_size=10)
        cprint(f"subscribe to {rospy.get_param('/robot/modified_state_sensor/topic', '/modified_state')}", self._logger)
        rospy.init_node('modified_state_frame_visualizer')

    def _publish_combined_global_poses(self, data: np.ndarray) -> None:
        resolution = (848, 800)
        # position, width, height = calculate_bounding_box(state=data,  # TODO
        #                                                  resolution=resolution)
        position, width, height = (10, 10), 5, 5
        frame = np.zeros(resolution)
        frame[position[0]:position[0] + width,
        position[1]:position[1] + height] = 255

        image = Image()
        image.data = frame.astype(np.uint8).flatten().tolist()
        image.height = resolution[0]
        image.width = resolution[1]
        image.encoding = 'mono8'
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



