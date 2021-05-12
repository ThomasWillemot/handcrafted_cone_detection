#!/usr/bin/python3.8
"""Extract waypoints out of a single image.
    Has two functionolities
        Image retrieval: calculates 3d location based on an image
        Service for the controller which answers with a 3d location
"""
import numpy as np
from cv_bridge import CvBridge
import rospy
import cv2

from sensor_msgs.msg import *
from tf2_msgs.msg import *
from std_msgs.msg import *
from geometry_msgs import *
from std_srvs.srv import Trigger
from handcrafted_cone_detection.srv import SendRelCor, SendRelCorResponse
from handcrafted_cone_detection.msg import ConeImgLoc


class WaypointExtractor:

    def __init__(self):

        rospy.init_node('waypoint_extractor_server')
        self.bridge = CvBridge()
        #dim = (848, 800)
        self.threshold = 95
        #k = np.array(
        #    [[285.95001220703125, 0.0, 418.948486328125], [0.0, 286.0592956542969, 405.756103515625], [0.0, 0.0, 1.0]])
        #d = np.array(
        #    [[-0.006003059912472963], [0.04132957011461258], [-0.038822319358587265], [0.006561396177858114]])
        #self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, dim,
        #                                                           cv2.CV_16SC2)
        self.counter = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.mask = cv2.imread('src/sim/ros/python3_ros_ws/src/handcrafted_cone_detection/src/frame_drone_mask.png', 0)
        self.kernel = np.ones((3, 3), np.uint8)
        self.x_array_med = [0, 0, 0, 0, 0]
        self.y_array_med = [0, 0, 0, 0, 0]
        self.z_array_med = [0, 0, 0, 0, 0]
        self.last_idx = 0

        self.rate = rospy.Rate(1000)
        self.image1_buffer = []
        self.image2_buffer = []
        self.image_stamp = rospy.Time(0)
        #TODO enable if on drone self._init_fsm_handshake_srv()
        self.pub = rospy.Publisher('cone_coordin', ConeImgLoc, queue_size=10)
        self.thresh_pub = rospy.Publisher('threshold_im', Image, queue_size=10)
    # Function to extract the cone out of an image. The part of the cone(s) are binary ones, the other parts are 0.
    # inputs: image and color of cone
    # output: binary of cone
    def get_cone_binary(self, current_image, threshold):
        binary_image = cv2.threshold(current_image, threshold, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        return binary_image[1]

    # Extract the 2d location in the image after segmentation.
    def get_cone_2d_location(self, bin_im, left):
        plot_im = bin_im
        row_sum = np.sum(bin_im, axis=1)
        for row_idx in range(799):
            if row_sum[row_idx] > 400 * 255:
                airrow = row_idx
        bin_im[0:airrow, :] = 0
        i = airrow
        prev_empty = False
        while i < 799:
            curr_empty = row_sum[i] > 255
            if curr_empty:
                bin_im[i, :] = np.zeros(848)
            elif prev_empty:
                break
            else:
                prev_empty = True
            i += 1
        filtered_np_gray = cv2.morphologyEx(bin_im, cv2.MORPH_OPEN, self.kernel)
        img_masked = cv2.bitwise_and(filtered_np_gray, self.mask)
        row_sum = np.sum(img_masked, axis=1)
        cone_found = False
        cone_row = 0
        max_row = 0
        row = 799  # start where no drone parts are visible in image
        cone_started = False
        while not cone_found and row >= 0:
            if row_sum[row] >= max_row and row_sum[row] > 4*255:
                cone_row = row
                max_row = row_sum[row]
                cone_started = True
            elif cone_started:
                cone_found = True
            row -= 1

        current_start = 0
        max_start = 0
        max_width = 0
        current_width = 0
        for col_index in range(847):
            if img_masked[cone_row, col_index] == 0:
                if current_width > max_width:
                    max_width = current_width
                    max_start = current_start
                current_width = 0
                current_start = 0
            else:
                if current_start == 0:
                    current_start = col_index
                current_width += 1
        if left:
            self.image_publisher(max_start, cone_row, max_width)
            self.threshol_image_publish(plot_im, max_start, cone_row, max_width)
        return [max_start + int(np.ceil(max_width / 2)) - 424, -cone_row + 400, max_width]

    def get_depth_triang(self, im_coor_1, im_coor_2):
        x_fish1 = im_coor_1[0]
        x_fish2 = im_coor_2[0]
        y_fish1 = im_coor_1[1]
        y_fish2 = im_coor_2[1]
        baseline = 0.064  # 6.4mm???
        disparity = x_fish1 - x_fish2
        if disparity == 0:
            disparity = 1
        x = baseline * x_fish1 / disparity
        y = baseline * y_fish1 / disparity
        z = baseline * 286 / disparity
        x_cor = z
        y_cor = -x
        z_cor = y
        return np.array([x_cor, y_cor, z_cor])

    # Extracts the waypoints (3d location) out of the current image.
    def extract_waypoint(self, image, left):
        print("Extract wp")
        cv_im = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')  # Load images to cv

        rect_image = cv2.remap(cv_im, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)  # Remap fisheye to normal picture
        cut_off = 335
        #rect_image[848 - cut_off:848, :] = 0  # set the drone frame as zeros. Should not be detected as cone.

        # Cone segmentation
        bin_im = self.get_cone_binary(rect_image, threshold=self.threshold)

        # Positioning in 2D of cone parts
        loc_2d = self.get_cone_2d_location(bin_im, left)
        return loc_2d

    # update the median filter with length 3
    def update_median(self, coor):
        self.x_array_med[self.last_idx] = coor[0]
        self.x = np.median(self.x_array_med)
        self.y_array_med[self.last_idx] = coor[1]
        self.y = np.median(self.y_array_med)
        self.z_array_med[self.last_idx] = coor[2]
        self.z = np.median(self.z_array_med)
        self.last_idx += 1
        if self.last_idx == 5:
            self.last_idx = 0

    # Handles the service requests.
    def handle_cor_req(self, req):
        print("STAMP")
        print(self.image_stamp)
        # TEST DUMMY - REMOVE THIS
        # coor = [0, 3, 4]
        return SendRelCorResponse(self.x, self.y, self.z, self.image_stamp)

    def _init_fsm_handshake_srv(self):
        """Setup handshake service for FSM.
        """
        self.fsm_handshake_srv = rospy.Service(
            "/waypoint_extractor_server/fsm_handshake", Trigger, self.fsm_handshake)

    def fsm_handshake(self, _):
        '''Handles handshake with FSM. Return that initialization was successful and 
	    waypoint exctractor is running.
        '''
        return {"success": True, "message": ""}

    def rel_cor_server(self):
        '''Service for delivery of current relative coordinates
        '''
        s = rospy.Service('/waypoint_extractor_server/rel_cor', SendRelCor, self.handle_cor_req)
        rospy.loginfo("WPE  - Waypoint extractor running. Waiting for request")

    def image_subscriber(self):
        '''Subscribes to topics and and runs callbacks
        '''
        # These always come with identical timestamps. Callbacks at slightly offset times.
        rospy.Subscriber("/camera/fisheye1/image_raw", Image, self.fisheye1_callback)
        rospy.Subscriber("/camera/fisheye2/image_raw", Image, self.fisheye2_callback)

    def fisheye1_callback(self, image):
        '''Buffer images coming from /camera/fisheye1/image_raw. Buffer is cleared in run().
        Args:
            image: std_msgs/Image
        '''
        self.image1_buffer.append(image)

    def fisheye2_callback(self, image):
        '''Buffer images coming from /camera/fisheye2/image_raw. Buffer is cleared in run().
        Args:
            image: std_msgs/Image
        '''
        self.image2_buffer.append(image)

    def threshol_image_publish(self,image,max_start,cone_row,max_width):
        resolution = (800, 848)
        frame = np.array(image)
        frame = cv2.circle(frame, (max_start + int(max_width/2), cone_row), int(max_width/2), 255, 5)
        image = Image()
        image.data = frame.astype(np.uint8).flatten().tolist()
        image.height = resolution[0]
        image.width = resolution[1]
        image.encoding = 'mono8'
        image.step = resolution[1]
        self.thresh_pub.publish(image)

    def image_publisher(self, x_coor, y_coor, width):
        cone_coor1 = ConeImgLoc()
        cone_coor1.x_pos = np.int32(x_coor+int(width/2))
        cone_coor1.y_pos = np.int32(y_coor)
        cone_coor1.cone_width = np.int16(width)
        self.pub.publish(cone_coor1)

    def run(self):
        '''Starts all needed functionalities + Main loop
        '''
        self.image_subscriber()
        self.rel_cor_server()

        while not rospy.is_shutdown():
            if self.image1_buffer and self.image2_buffer:
                image1 = self.image1_buffer.pop()
                # print("pop")
                # print(image1.header.stamp.to_sec())

                self.image1_buffer.clear()
                self.image2_buffer.clear()

                relat_coor = self.extract_waypoint(image1)
                print('Coordinates')
                print(round(self.running_average[0], 2), round(self.running_average[1], 2),
                      round(self.running_average[2], 2))
                self.publish_reference(self.running_average)
                self.image_stamp = image1.header.stamp

            self.rate.sleep()

        # rospy.spin()


if __name__ == "__main__":
    waypoint_extractor = WaypointExtractor()
    waypoint_extractor.run()
