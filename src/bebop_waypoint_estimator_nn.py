#!/usr/bin/python3.8
"""Extract waypoints out of a single image.
    Has two functionolities
        Image retrieval: calculates 3d location based on an image
        Service for the controller which answers with a 3d location
"""
import os
import time
from glob import glob

import numpy as np
import torch
from cv_bridge import CvBridge
import rospy
import cv2
from geometry_msgs.msg import PointStamped, Point

from sensor_msgs.msg import *
from tf2_msgs.msg import *
from std_msgs.msg import *
from geometry_msgs import *
from std_srvs.srv import Trigger
from handcrafted_cone_detection.srv import SendRelCor, SendRelCorResponse
from handcrafted_cone_detection.msg import ConeImgLoc
from src.sim.ros.python3_ros_ws.src.handcrafted_cone_detection.helper_files import cnn_architecture, bebop_400_arch_2
from src.sim.ros.python3_ros_ws.src.handcrafted_cone_detection.helper_files.ArchitectureConfig import ArchitectureConfig


class WaypointEstimatorNN:

    def __init__(self, safe_flight=True):
        rospy.init_node('waypoint_extractor_server')
        self.bridge = CvBridge()
        self.total_time = time.time()
        dim = (856, 480)
        self.safe_flight = safe_flight
        self.threshold = 210  # TODO change if needed using rviz (check images)
        # TODO !
        k = np.array([[537.292878, 0.0, 427.331854], [0.0, 527.000348, 240.226888], [0.0, 0.0, 1.0]])
        d = np.array([[0.004974], [-0.00013], [-0.001212], [0.002192]])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, dim,
                                                                   cv2.CV_16SC2)
        self.mask = cv2.imread('src/sim/ros/python3_ros_ws/src/handcrafted_cone_detection/src/frame_drone_mask.png', 0)
        # TODO change path of mask
        self.counter = 0
        self.av_fps = .01
        self.kernel = np.ones((5, 5), np.uint8)
        self.rate = rospy.Rate(1000)
        self.image1_buffer = []
        self.image2_buffer = []
        self.image_stamp = rospy.Time(0)
        self.running_average = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.original_model_device = 'default'
        # TODO enable if on drone self._init_fsm_handshake_srv()
        self.pub = rospy.Publisher('cone_coordin', ConeImgLoc, queue_size=10)
        self.reference_publisher = rospy.Publisher('reference_pose', PointStamped, queue_size=10)
        self.thresh_pub = rospy.Publisher('threshold_im', Image, queue_size=10)
        self.bin_im_publisher = rospy.Publisher('bin_image', Image, queue_size=10)
        architecture_config = ArchitectureConfig()
        self.trainer = None
        self.environment = None
        self.epoch = 0
        self.net = eval('cnn_architecture').Net(config=architecture_config) \
            if architecture_config is not None else None
        # cnn_architecture: bebop_400_arch_2
        self.load_checkpoint('/media/thomas/Elements/training_nn/res_200/6100_lr_0002')
        #self.load_checkpoint('/media/thomas/Elements/training_nn/bebop_high_res/net_str_1_new_ARCH_5')
        self.put_model_on_device('cuda')

    # Function to extract the cone out of an image. The part of the cone(s) are binary ones, the other parts are 0.
    # inputs: image and color of cone
    # output: binary of cone
    def get_cone_binary(self, current_image, threshold):
        binary_image = cv2.threshold(current_image, threshold, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        return binary_image[1]

    def put_model_on_device(self, device: str = None):
        original_model_device = self.net.get_device()
        torch.device(device)

    def put_model_back_to_original_device(self):
        self.net.set_device(self.original_model_device)

    def downsample_image(self, image, factor=1):
        img = np.array(image, dtype='float32')
        img = torch.from_numpy(img.reshape(1, 1, img.shape[0], img.shape[1]))  # Convert grayscale image to tensor
        maxPool = torch.nn.AvgPool2d(factor)  # 4*4 window, maximum pooling with a step size of 4
        img_tensor = maxPool(img)
        img = torch.squeeze(img)  # Remove the dimension of 1
        img = img.numpy().astype('uint8')  # Conversion format, ready to output
        return img_tensor, img

    def extend_image(self, image, dimensions):
        empty_image = np.zeros(dimensions)
        orig_shape = image.shape
        empty_image[int(dimensions[0] / 2 - orig_shape[0] / 2):int(dimensions[0] / 2 + orig_shape[0] / 2),
                      :] = image[:,4:852]
        return empty_image

    def crop_image(self, image, dimensions):
        orig_shape = image.shape
        print(orig_shape[0] / 2 - dimensions[0] / 2)
        return image[int(orig_shape[0] / 2 - dimensions[0] / 2):int(orig_shape[0] / 2 + dimensions[0] / 2),
               int(orig_shape[1] / 2 - dimensions[1] / 2):int(orig_shape[1] / 2 + dimensions[1] / 2)]

    # Extracts the waypoints (3d location) out of the current image.
    def extract_waypoint(self, image,NN = True):
        cv_im = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')  # Load images to cv

        #rect_image = cv2.remap(cv_im, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
        #                      borderMode=cv2.BORDER_CONSTANT)  # Remap fisheye to normal picture #TODO is already rectified on bebop

        # Cone segmentation

        post_proc_or = self.post_process_yellow_cones(cv_im)
        post_proc_im = self.post_process_image(post_proc_or)
        cropped_image = self.extend_image(post_proc_im, [800, 848])  # TODO horizon

        if NN:
            down_tens_image, img = self.downsample_image(cropped_image, factor=4)
            #bin_im_sum = np.sum(np.sum(img, axis=1), axis=0)
            # Positioning in 2D of cone parts
            cone_coordinates = self.eval_neural_net(down_tens_image)
            # Use scaling factor ( neural net is trained in other way than images are.
            cone_coordinates *= .8
            self.bin_im_publish(255 * img)
            #if bin_im_sum > 400 * 400:
            #    print("not used")
            #    cone_coordinates = np.array([0, 0, 0, 0, 0, 0])
            #self.running_average = self.running_average * 0.65 + cone_coordinates * 0.35
            #    cone_coordinates = self.running_average
        else:
            tune_factor = 420

            image_coord = self.get_cone_2d_location(255*cropped_image)
            self.total_time = time.time()
            cone_coordinates_first = self.get_cone_3d_location(image_coord[2], 0.18, image_coord[0:2], tune_factor)
            time_rect = (time.time() - self.total_time)
            self.av_fps = time_rect * 0.01 + self.av_fps * 0.99
            print(self.av_fps)
            cone_coordinates = np.array([cone_coordinates_first[0],cone_coordinates_first[1],cone_coordinates_first[2],2,1,1])
            self.bin_im_publish(cropped_image*255)


        if False:
            k = k
        else:
            x_position = int(-cone_coordinates[1] / cone_coordinates[0] * 539 + 427)  # TODO
            y_position = int(-cone_coordinates[2] / cone_coordinates[0] * 527 + 240)  # TODO
            x_2_position = int(-cone_coordinates[4] / cone_coordinates[3] * 539 + 427)  # TODO
            y_2_position = int(-cone_coordinates[5] / cone_coordinates[3] * 529 + 239)  # TODO
            #cone_coordinates[0] -= 2.5
            self.running_average = self.running_average * 0.65 + cone_coordinates * 0.35
            cone_coordinates = self.running_average
            #self.image_publisher(x_position, y_position, 25 / cone_coordinates[0])
            self.threshold_image_publish(cv_im, x_position, y_position, 150 / 2, x_2=x_2_position, y_2=y_2_position,
                                         size_2=150 / cone_coordinates[3])

        return cone_coordinates

    # Used to make bin map of orange cones
    def post_process_orange_cones(self, image):
        #res_rg = image[:, :, 0]  - image[:, :, 2]//4 - image[:,:,1]//4
        th_val, th_r = cv2.threshold(image[:, :, 0], 225, 1, cv2.THRESH_BINARY)
        img_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        th_val, th_g = cv2.threshold(image[:, :, 1], 200, 1, cv2.THRESH_BINARY)
        th_val, th_b = cv2.threshold(image[:, :, 2], 240, 1, cv2.THRESH_BINARY)
        not_th_b = cv2.bitwise_not(th_b)
        not_th_g = cv2.bitwise_not(th_g)
        r_not_b = cv2.bitwise_and(not_th_b,th_r)
        r_nog_b_not_g = cv2.bitwise_not(r_not_b,not_th_g)
        r_nog_b_not_g = cv2.morphologyEx(r_nog_b_not_g, cv2.MORPH_OPEN, self.kernel)
        return r_not_b

    # Used to make bin map of orange cones
    def post_process_yellow_cones(self, image):
        img_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        th_val, th_bw = cv2.threshold(img_bw, 180, 1, cv2.THRESH_BINARY)
        filtered_np_bw = cv2.morphologyEx(th_bw, cv2.MORPH_OPEN, self.kernel)
        return filtered_np_bw

    def and_of_images(self, image1, image2):
        return cv2.bitwise_and(image1, image2)

    def eval_neural_net(self, image):
        predictions = self.net.forward(image, train=False)
        np_pred = predictions.detach().numpy()
        return np_pred[0]

    def load_checkpoint(self, checkpoint_dir: str):
        if not checkpoint_dir.endswith('torch_checkpoints'):
            checkpoint_dir += '/torch_checkpoints'
        if len(glob(f'{checkpoint_dir}/*.ckpt')) == 0 and len(glob(f'{checkpoint_dir}/torch_checkpoints/*.ckpt')) == 0:
            raise FileNotFoundError
        # Get checkpoint in following order
        if os.path.isfile(os.path.join(checkpoint_dir, 'checkpoint_best.ckpt')):
            checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_best.ckpt')
        elif os.path.isfile(os.path.join(checkpoint_dir, 'checkpoint_latest.ckpt')):
            checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_latest.ckpt')
        else:
            checkpoints = {int(f.split('.')[0].split('_')[-1]): os.path.join(checkpoint_dir, f)
                           for f in os.listdir(checkpoint_dir)}
            checkpoint_file = checkpoints[max(checkpoints.keys())]
        # Load params for each experiment element
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else 0
        for element, key in zip([self.net, self.trainer, self.environment],
                                ['net_ckpt', 'trainer_ckpt', 'environment_ckpt']):
            if element is not None and key in checkpoint.keys():
                element.load_checkpoint(checkpoint[key])
        print('checkpoint loaded')

    def post_process_image(self, image, binary=False):
        height = 480  # TODO
        width = 856
        row_sum = np.sum(image, axis=1)  # should be 800 high
        airrow = 0
        for row_idx in range(height):
            if row_sum[row_idx] > height / 2 * 1:  # TODO
                airrow = row_idx
        image[0:airrow, :] = 0
        i = airrow
        prev_empty = False
        while i < height - 1:
            curr_empty = row_sum[i] > 4 * 1
            if curr_empty:
                image[i, :] = 0
            elif prev_empty:
                break
            else:
                prev_empty = True
            i += 1
        image_np_gray = np.asarray(image)
        if np.amax(image_np_gray) == 255:
            image_np_gray = image_np_gray / 255
        return image_np_gray

    def get_cone_2d_location(self, bin_im):
        row_sum = np.sum(bin_im, axis=1)
        i = 0

        while row_sum[i] > 1 and i < 799:
            bin_im[i, :] = np.zeros(848)
            i += 1

        airrow = 0
        for row_idx in range(799):
            if row_sum[row_idx] > 400 * 255:
                airrow = row_idx
        bin_im[1:airrow, :] = 0
        row_sum = np.sum(bin_im, axis=1)
        cone_found = False
        cone_row = 0
        max_row = 0
        row = 799  # start where no drone parts are visible in image
        cone_started = False
        while not cone_found and row >= 0:
            if row_sum[row] >= max_row and row_sum[row] > 4 * 255:
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
            if bin_im[cone_row, col_index] == 0:
                if current_width > max_width:
                    max_width = current_width
                    max_start = current_start
                current_width = 0
                current_start = 0
            else:
                if current_width == 0:
                    current_start = col_index
                current_width += 1
        if current_width > max_width:
            max_width = current_width
            max_start = current_start

        return [max_start + int(np.ceil(max_width / 2)) - 424, -cone_row + 400, max_width]

    def get_cone_3d_location(self, cone_width_px, cone_width_m, conetop_coor, tune_factor):
        x_cor = 0
        y_cor = 0
        z_cor = -1  # do not update if z remains -1 TODO
        if cone_width_px > 0:  # only updates when cone detected
            # position relative to the camera in meters.
            z_cor = cone_width_m * tune_factor / cone_width_px
            x_cor = conetop_coor[0] * z_cor / tune_factor
            y_cor = conetop_coor[1] * z_cor / tune_factor
        return np.array([z_cor, -x_cor, y_cor])

    # Handles the service requests.
    def handle_cor_req(self, req):
        print("STAMP")
        print(self.image_stamp)
        # TEST DUMMY - REMOVE THIS
        # coor = [0, 3, 4]
        return SendRelCorResponse(self.running_average[0], self.running_average[1], self.running_average[2],
                                  self.running_average[3], self.running_average[4], self.running_average[5],
                                  self.image_stamp)

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
        rospy.Subscriber("/bebop/image_raw", Image, self.fisheye1_callback)

    def fisheye1_callback(self, image):
        '''Buffer images coming from /camera/fisheye1/image_raw. Buffer is cleared in run().
        Args:
            image: std_msgs/Image
        '''
        self.image1_buffer.append(image)

    '''Augments the grayscale or binary image
    Args:
        image: bin numpy image
        max_start: image coordinate u for the circle
        cone_row: image coordinate v for the circle
        max_width: width of the circle
    '''

    def threshold_image_publish(self, image, x, y, size, x_2, y_2, size_2):
        resolution = image.shape
        frame = np.array(image)
        frame = cv2.circle(frame, (x, y), int(max(size, 2) / 2), (255, 0, 0), 2)  # 255
        frame = cv2.circle(frame, (x_2, y_2), int(max(size_2, 2) / 2), (255, 0, 0), 2)
        image = Image()
        image.data = frame.astype(np.uint8).flatten().tolist()
        image.height = resolution[0]
        image.width = resolution[1]
        image.encoding = 'rgb8'  # 'mono8'
        image.step = resolution[1]
        self.thresh_pub.publish(image)

    def bin_im_publish(self, image):
        resolution = image.shape
        frame = np.array(image)
        image = Image()
        image.data = frame.astype(np.uint8).flatten().tolist()
        image.height = resolution[0]
        image.width = resolution[1]
        image.encoding = 'mono8'
        image.step = resolution[1]
        self.bin_im_publisher.publish(image)

    # publishes coordiantes.
    def image_publisher(self, x_coor, y_coor, width):
        cone_coor1 = ConeImgLoc()
        cone_coor1.x_pos = np.int32(x_coor)
        cone_coor1.y_pos = np.int32(y_coor)
        cone_coor1.cone_width = np.int16(int(max(width, 2)))
        self.pub.publish(cone_coor1)

    # Publish the drone location with an offset height.
    def publish_reference(self, coordinates):
        cam_angle = 2*np.pi/360*30
        if self.safe_flight:
            coordinates = coordinates
        x_glob = coordinates[0]*np.cos(cam_angle) + coordinates[2]*np.sin(cam_angle)
        z_glob = -coordinates[0]*np.sin(cam_angle) + coordinates[2]*np.cos(cam_angle) +1
        z_glob = 0
        ref = PointStamped(header=Header(frame_id="agent"),
                           point=Point(x=x_glob, y=coordinates[1], z=z_glob))
        #print('REFERENTIE')
        #print(ref)
        self.reference_publisher.publish(ref)

    def run(self):
        '''Starts all needed functionalities + Main loop
        '''
        self.image_subscriber()
        self.rel_cor_server()

        while not rospy.is_shutdown():
            if self.image1_buffer:
                image1 = self.image1_buffer.pop()
                # print("pop")
                # print(image1.header.stamp.to_sec())

                self.image1_buffer.clear()
                self.image2_buffer.clear()

                relat_coor = self.extract_waypoint(image1,NN=False)

                #self.total_time = time.time()
                print('Coordinates')
                print(round(self.running_average[0], 2), round(self.running_average[1], 2),
                      round(self.running_average[2], 2))

                self.publish_reference(self.running_average)
                self.image_stamp = image1.header.stamp
            self.rate.sleep()

        # rospy.spin()


if __name__ == "__main__":
    waypoint_estimator_nn = WaypointEstimatorNN(safe_flight=False)
    waypoint_estimator_nn.run()
