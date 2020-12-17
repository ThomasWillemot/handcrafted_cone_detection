#!/usr/bin/python3
"""Extract waypoints out of a single image.
    Has two functionolities
        Image retrieval: calculates 3d location based on an image
        Service for the controller which answers with a 3d location
"""
import numpy as np
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import *
from tf2_msgs.msg import *
from handcrafted_cone_detection.srv import SendRelCor, SendRelCorResponse
import cv2
from std_msgs.msg import *
from geometry_msgs import *
from std_srvs.srv import Trigger


class WaypointExtractor:

    def __init__(self):

        rospy.init_node('waypoint_extractor_server')

        self.bridge = CvBridge()
        dim = (800, 848)
        k = np.array(
            [[285.95001220703125, 0.0, 418.948486328125], [0.0, 286.0592956542969, 405.756103515625], [0.0, 0.0, 1.0]])
        d = np.array(
            [[-0.006003059912472963], [0.04132957011461258], [-0.038822319358587265], [0.006561396177858114]])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, dim,
                                                                   cv2.CV_16SC2)
        self.counter = 0
        self.x_1 = -1
        self.y_1 = -1
        self.x_2 = 0
        self.y_2 = 0
        self.x_orig = 0
        self.y_orig = 0
        self.z_orig = 0
        self.imagexite_rec = 0
        self.rate = rospy.Rate(1000)
        self.is_processing_image1 = False
        self.is_processing_image2 = False
        self.image1_buffer = []
        self.image2_buffer = []


    # Function to extract the cone out of an image. The part of the cone(s) are binary ones, the other parts are 0.
    # inputs: image and color of cone
    # output: binary of cone
    def get_cone_binary(self, current_image, threshold):
        binary_image = cv2.threshold(current_image, threshold, 255, cv2.THRESH_BINARY)
        return binary_image[1]

    # Extract the 2d location in the image after segmentation.
    # TODO: loops of get_cone_binary and this function can be written in one.
    def get_cone_2d_location(self, bin_im):
        cone_found = False
        im_size = bin_im.shape
        max_width_row = 0
        max_width_x = -1
        max_width_y = -1
        current_width_start = -1
        current_width = 0
        prev_pix = 0
        row = im_size[0] - 1
        prev_row = 0
        while not cone_found and row >= 0:
            for column in range(im_size[1]):
                if bin_im[row, column] > 0:
                    if current_width_start == -1:
                        current_width_start = column
                    elif prev_pix == 1:
                        current_width += 1
                    elif current_width == 0:
                        current_width_start = column
                    prev_pix = 1
                else:
                    prev_pix = 0
            if current_width > max_width_row and current_width_start > 0:
                max_width_row = current_width
                max_width_x = current_width_start
                max_width_y = row
            if prev_row == 1 and current_width_start == -1 and max_width_row > 2:
                cone_found = True
            if current_width_start > -1:
                prev_row = 1
            current_width = 0
            current_width_start = -1
            row -= 1
        max_width_row += 1
        return [max_width_x - 400 + int(np.ceil(max_width_row / 2)), -max_width_y + 424,
                max_width_row]  # counting starts at zero

    def get_cone_2d_speedup(self, bin_im, prev_coor):
        if prev_coor[2] == 0:
            return self.get_cone_2d_strided(bin_im)
        x_prev_pix = prev_coor[0] + 400
        y_prev_pix = -prev_coor[1] + 424
        if prev_coor[1] > 846 or prev_coor[0] <2 or prev_coor[1] <2 or prev_coor[0] >= 799:
            return self.get_cone_2d_strided(bin_im)
        return self.search_diag(bin_im, x_prev_pix, y_prev_pix)

    def search_left_up(self, bin_im, x_prev_pix, y_prev_pix):
        if y_prev_pix == 847 or x_prev_pix == 0 or y_prev_pix == 0 or x_prev_pix == 799:
            return [x_prev_pix, y_prev_pix]
        if bin_im[y_prev_pix + 1, x_prev_pix - 1] > 0:  # left
            return self.search_left_down(bin_im, x_prev_pix - 1, y_prev_pix + 1)
        elif bin_im[y_prev_pix - 1, x_prev_pix - 1] > 0:  # down
            return self.search_left_up(bin_im, x_prev_pix - 1, y_prev_pix - 1)
        else:
            return [x_prev_pix, y_prev_pix]

    def search_right_up(self, bin_im, x_prev_pix, y_prev_pix):
        if y_prev_pix == 847 or x_prev_pix == 0 or y_prev_pix == 0 or x_prev_pix == 799:
            return [x_prev_pix, y_prev_pix]
        if bin_im[y_prev_pix + 1, x_prev_pix + 1] > 0:  # left
            return self.search_right_down(bin_im, x_prev_pix + 1, y_prev_pix + 1)
        elif bin_im[y_prev_pix - 1, x_prev_pix + 1] > 0:  # up
            return self.search_right_up(bin_im, x_prev_pix + 1, y_prev_pix - 1)
        else:
            return [x_prev_pix, y_prev_pix]

    def search_left_down(self, bin_im, x_prev_pix, y_prev_pix):
        if y_prev_pix == 847 or x_prev_pix == 0 or y_prev_pix == 0 or x_prev_pix == 799:
            return [x_prev_pix, y_prev_pix]
        if bin_im[y_prev_pix - 1, x_prev_pix - 1] > 0:  # right
            return self.search_left_up(bin_im, x_prev_pix - 1, y_prev_pix - 1)
        elif bin_im[y_prev_pix + 1, x_prev_pix - 1] > 0:  # down
            return self.search_left_down(bin_im, x_prev_pix - 1, y_prev_pix + 1)
        else:
            return [x_prev_pix, y_prev_pix]

    def search_right_down(self, bin_im, x_prev_pix, y_prev_pix):
        if y_prev_pix == 847 or x_prev_pix == 0 or y_prev_pix == 0 or x_prev_pix == 799:
            return [x_prev_pix, y_prev_pix]
        if bin_im[y_prev_pix - 1, x_prev_pix + 1] > 0:  # right
            return self.search_right_up(bin_im, x_prev_pix + 1, y_prev_pix - 1)
        elif bin_im[y_prev_pix + 1, x_prev_pix + 1] > 0:  # up
            return self.search_right_down(bin_im, x_prev_pix + 1, y_prev_pix + 1)
        else:
            return [x_prev_pix, y_prev_pix]

    def search_diag(self, bin_im, prev_pix_x, prev_pix_y):
        # up left
        left_coor = [0, 0]
        right_coor = [0, 0]
        up_coor = [0, 0]
        down_coor = [0, 0]
        if bin_im[prev_pix_y - 1, prev_pix_x - 1]:
            up_coor = self.search_left_up(bin_im, prev_pix_x - 1, prev_pix_y - 1)
        # right up
        if bin_im[prev_pix_y - 1, prev_pix_x + 1]:
            right_coor = self.search_right_up(bin_im, prev_pix_x + 1, prev_pix_y - 1)
        # left down
        if bin_im[prev_pix_y + 1, prev_pix_x - 1]:
            left_coor = self.search_right_down(bin_im, prev_pix_x - 1, prev_pix_y + 1)
        # down right
        if bin_im[prev_pix_y + 1, prev_pix_x + 1]:
            down_coor = self.search_right_down(bin_im, prev_pix_x + 1, prev_pix_y + 1)
        left_side_obj = min(left_coor[0], up_coor[0])
        top_side_obj = min(left_coor[1], up_coor[1])
        right_side_obj = max(down_coor[0], right_coor[0])
        down_side_obj = max(down_coor[1], right_coor[1])

        if left_side_obj == right_side_obj:
            return self.get_cone_2d_strided(bin_im)
        else:
            return [int(np.ceil((left_side_obj + right_side_obj) / 2) - 400),
                    int(-1 * (np.ceil(top_side_obj + down_side_obj) / 2) + 424),
                    right_side_obj - left_side_obj]

    def get_cone_2d_strided(self, bin_im):
        cone_found = False
        im_size = bin_im.shape
        max_width_row = 0
        max_width_x = -1
        max_width_y = -1
        current_width_start = -1
        current_width = 0
        prev_pix = 0
        row = im_size[0] - 1
        prev_row = 0
        while not cone_found and row >= 0:
            for column in range(int(im_size[1] / 2)):
                if bin_im[row, column * 2] > 0:
                    if current_width_start == -1:
                        current_width_start = column * 2
                    elif prev_pix == 1:
                        current_width += 1
                    elif current_width == 0:
                        current_width_start = column * 2
                    prev_pix = 1
                else:
                    prev_pix = 0
            if current_width > max_width_row and current_width_start > 0:
                max_width_row = current_width
                max_width_x = current_width_start
                max_width_y = row
            if prev_row == 1 and current_width_start == -1 and max_width_row > 2:
                cone_found = True
            if current_width_start > -1:
                prev_row = 1
            current_width = 0
            current_width_start = -1
            row -= 2
        max_width_row = max_width_row * 2

        return [max_width_x - 400 + int(np.ceil(max_width_row / 2)), -max_width_y + 424,
                max_width_row]  # counting starts at zero

    def get_depth_triang(self, x_fish1, x_fish2, y_fish1, y_fish2):
        baseline = 0.064  # 6.4mm???
        disparity = x_fish1 - x_fish2
        if disparity == 0:
            disparity = 1
        x = baseline * x_fish1 / disparity
        y = baseline * y_fish1 / disparity
        z = baseline * 286 / disparity
        return [z, -x, y] #order of axis and right dimensions

    # Extracts the waypoints (3d location) out of the current image.
    def extract_waypoint_1(self, image):
        print("extract wp 1")
        cv_im = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')  # Load images to cv
        #current_image = cv2.remap(cv_im, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
        #                          borderMode=cv2.BORDER_CONSTANT)  # Remap fisheye to normal picture
        # Cone segmentation
        bin_im = self.get_cone_binary(cv_im, threshold=90)
        # Positioning in 2D of cone parts
        loc_2d = self.get_cone_2d_location(bin_im)
        # Get the position and width of the cone
        max_width = loc_2d[2]  # Max width detected, assumption biggest object is the cone.
        # Index of middle of max width
        # Remap 2D locations to 3D using width
        self.x_1 = loc_2d[0]
        self.y_1 = loc_2d[1]

    def extract_waypoint_2(self, image):
        print("extract wp 2")
        cv_im = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')  # Load images to cv
        #current_image = cv2.remap(cv_im, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
        #                          borderMode=cv2.BORDER_CONSTANT)  # Remap fisheye to normal picture
        # Cone segmentation
        bin_im = self.get_cone_binary(cv_im, threshold=90)
        # Positioning in 2D of cone parts
        loc_2d = self.get_cone_2d_location(bin_im)
        # Get the position and width of the cone
        max_width = loc_2d[2]  # Max width detected, assumption biggest object is the cone.
        # Index of middle of max width
        # Remap 2D locations to 3D using width
        self.x_2 = loc_2d[0]
        self.y_2 = loc_2d[1]

    # Handles the service requests.
    def handle_cor_req(self, req):
        # should also transform last coordinates
        coor = self.get_depth_triang(self.x_1, self.x_2, self.y_1, self.y_2)
        if coor[0]>1:
            coor[0] =1
            coor[1] /=coor[0]
            coor[2] /= coor[0]
        elif coor[0]< 0:
            coor[0] = 0
            coor[1] = 0
            coor[2] = 0

        # TEST DUMMY - REMOVE THIS
        # coor = [0, 0, 10]
        return SendRelCorResponse(coor[0], coor[1], coor[2])
        # return SendRelCorResponse(self.x_orig, self.y_orig, self.z_orig)
    
    def _init_fsm_handshake_srv(self):
        '''
        '''
        self.fsm_handshake_srv = rospy.Service(
            "/waypoint_extractor_server/fsm_handshake", Trigger, self.fsm_handshake)

    def fsm_handshake(self, _):
        '''Handles handshake with FSM. Return that initialization was successful and 
	waypoint exctractor is running.
        '''
        return{"success": True, "message": ""}

    #  Service for delivery of current relative coordinates
    def rel_cor_server(self):
        s = rospy.Service('/waypoint_extractor_server/rel_cor', SendRelCor, self.handle_cor_req)
        rospy.loginfo("WPE  - Waypoint extractor running. Waiting for request")

    # Subscribes to topics and and runs callbacks
    def image_subscriber(self):
        # These always come together.
        rospy.Subscriber("/camera/fisheye1/image_raw", Image, self.fisheye1_callback)
        rospy.Subscriber("/camera/fisheye2/image_raw", Image, self.fisheye2_callback)

    def fisheye1_callback(self, image):
        self.image1_buffer.append(image)
        # print(image.header.stamp.to_sec())

    def fisheye2_callback(self, image):
        self.image2_buffer.append(image)
        # print(image.header.stamp.to_sec())


    # Starts all needed functionalities
    def run(self):
        self._init_fsm_handshake_srv()
        self.image_subscriber()
        self.rel_cor_server()
        
        while not rospy.is_shutdown():
             if self.image1_buffer and self.image2_buffer:
                 image1 = self.image1_buffer.pop()
                 image2 = self.image2_buffer.pop()
                 # Make sure that two processed images belong to same timestamp
                 if image1.header.stamp > image2.header.stamp:
                     while image1.header.stamp > image2.header.stamp:
                         image1 = self.image1_buffer.pop()
                 elif image1.header.stamp < image2.header.stamp:
                     while image1.header.stamp < image2.header.stamp:
                         image2 = self.image2_buffer.pop()
                 print("pop")
                 print(image1.header.stamp.to_sec())
                 print(image2.header.stamp.to_sec())
                 self.image1_buffer.clear()
                 self.image2_buffer.clear()

                 self.extract_waypoint_1(image1)
                 self.extract_waypoint_2(image2)
             
             # print(self.is_processing_image1, self.is_processing_image2)
             self.rate.sleep()
                 
        # rospy.spin()


if __name__ == "__main__":
    waypoint_extractor = WaypointExtractor()
    waypoint_extractor.run()
