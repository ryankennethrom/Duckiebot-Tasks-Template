#!/usr/bin/env python3

import os
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
import cv2
from cv_bridge import CvBridge
import time
import math

class PDCrossWalkMain(DTROS):

    def __init__(self, node_name, tasks, loopTasks=True, proportional_gain=0.0000002, derivative_gain=0.0000002, integral_gain=0.0000002, velocity=0.3, integral_saturation=500000):
        # initialize the DTROS parent class
        super(PDCrossWalkMain, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.tasks = tasks
        self.loopTasks = loopTasks

        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/undistorted_image/compressed"
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        
        # construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

        self.proportional_gain = proportional_gain
        self.derivate_gain = derivative_gain
        self.integral_gain = integral_gain
        self.vel = velocity
        self.integral_saturation = integral_saturation

        self._error = 20
        self._error_last = self._error
        self._integration_stored = 0

        self._blue_detection_topic = f"/{self._vehicle_name}/camera_node/blue_image_mask/compressed"
        self.blue_mask_sub = rospy.Subscriber(self._blue_detection_topic , CompressedImage, self.blue_detection_callback)

        self._yellow_detection_topic = f"/{self._vehicle_name}/camera_node/yellow_image_mask/compressed"
        self.yellow_mask_sub = rospy.Subscriber(self._yellow_detection_topic , CompressedImage, self.yellow_detection_callback)

        self._homography_yellow_mask_topic = f"/{self._vehicle_name}/camera_node/homography_yellow_mask/compressed"
        self._homography_yellow_mask_sub = rospy.Subscriber(self._homography_yellow_mask_topic, CompressedImage, self.homography_yellow_mask_callback)

        self._homography_blue_mask_topic = f"/{self._vehicle_name}/camera_node/homography_blue_mask/compressed"
        self._homography_blue_mask_sub = rospy.Subscriber(self._homography_blue_mask_topic, CompressedImage, self.homography_blue_mask_callback)

        self.blue_mask = None
        self.yellow_mask = None
        self.homography_yellow_mask = None
        self.homography_blue_mask = None

        self._tasks = tasks
        self._loop_tasks = loopTasks
    
    def homography_blue_mask_callback(self, msg):
        self.homography_blue_mask = self._bridge.compressed_imgmsg_to_cv2(msg)

    def homography_yellow_mask_callback(self, msg):
        self.homography_yellow_mask = self._bridge.compressed_imgmsg_to_cv2(msg)
        # _, x_coords = np.where(self.homography_yellow_mask > 0)

        # if len(x_coords) == 0:
        #     return

        # x_variance = np.var(x_coords)

        # magnitude = len(x_coords)

        # print(str(magnitude) + " " + str(x_variance))

    def yellow_detection_callback(self, msg):
        # msg is already a mask
        self.yellow_mask = self._bridge.compressed_imgmsg_to_cv2(msg)
    
    def blue_detection_callback(self, msg):
        # msg is already a mask
        self.blue_mask = self._bridge.compressed_imgmsg_to_cv2(msg)

        # _, x_coords = np.where(self.blue_mask> 0)

        # if len(x_coords) == 0:
        #     return

        # x_variance = np.var(x_coords)

        # magnitude = len(x_coords)

        # print(str(magnitude) + " " + str(x_variance))
    
    def callback(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        h, w, _ = image.shape

        img_size = (w, h)
        
        src = np.float32([
            [0,382],
            [224, 191],  # Bottom left (near where left lane line is)
            [589, 382],  # Bottom right (near where right lane line is)
            [364, 191],  # Top left (near vanishing point for left lane)
        ])

        dst = np.float32([
            [100, 382],
            [100, 0],  # Bottom left (destination for left lane)
            [489, 382],  # Bottom right (destination for right lane)
            [489, 0],    # Top left (destination after warping)
        ])

        # cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)  # Red dots

        M = cv2.getPerspectiveTransform(src, dst)
        
        warped = cv2.warpPerspective(image, M, img_size)

        # Convert warped image to HSV for color detection
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

        # Define HSV range for detecting **white color**
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 50, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Define HSV range for detecting **yellow color**
        lower_yellow = np.array([15, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        self._error = self.compute_error(mask=mask_yellow, target_x=100) + self.compute_error(mask=mask_white, target_x=489)
    
    def getUpdate(self):
        P = self._error*self.proportional_gain
        errorRateOfChange = self._error - self._error_last
        D = self.derivate_gain * errorRateOfChange
        integration_stored_update = self._integration_stored + (self._error)
        self._integration_stored = (integration_stored_update) if abs(integration_stored_update) <= self.integral_saturation else (integration_stored_update/integration_stored_update)*self.integral_saturation
        I = self.integral_gain * self._integration_stored

        self._error_last = self._error

        return P + I + D

    def run(self):
        if self._loop_tasks == True:
            counter = 0
            while not rospy.is_shutdown():
                self._distance_right = 0
                self._distance_left = 0
                self._tasks[counter%len(self._tasks)].execute(self)
                counter+=1
        else:
            for task in self._tasks:
                
                self._distance_right = 0
                self._distance_left = 0

                task.execute(self)
            
        rospy.signal_shutdown(reason="tasks complete")
    
    def compute_error(self, mask, target_x=100, pixel_value=1):
        """
        Computes the sum of errors between detected yellow lane pixels and the expected lane position at x=100.
        
        Args:
            mask_yellow (numpy array): Binary image where white pixels represent detected yellow lane.
            target_x (int): Expected x-coordinate of the lane.

        Returns:
            float: The sum of errors between detected lane pixels and the expected position at x=100.
        """
        # Find nonzero (active) pixel coordinates
        y_coords, x_coords = np.where(mask > 0)  # y, x positions of active pixels
        
        if len(x_coords) == 0:
            return 0  # No detected yellow pixels, return 0 error

        # Compute the error as the difference from target_x
        errors = (x_coords - target_x) * pixel_value

        # Return the sum of the errors
        return np.sum(errors)
    
    def on_shutdown(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop)

class PDCrossWalkTask():
    def execute(self, dtros):
        self.runTask(dtros)
    
    def runTask(self, dtros):
        raise Exception("runTask must be overriden for all classes inheriting Task")

class FindCrossWalk(PDCrossWalkTask):

    def __init__(self, detection_magnitude=2000, stopping_distance_in_pixels=5):
        super().__init__()

        self.detection_magnitude = detection_magnitude
        self.stopping_distance_in_pixels = stopping_distance_in_pixels

    def runTask(self, dtros):
        print("Finding crosswalk")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            correctionUpdate = dtros.getUpdate()

            if dtros.homography_blue_mask is not None and self.getCenterDistanceFromBottom(dtros.homography_blue_mask) <= self.stopping_distance_in_pixels:
                break

            if correctionUpdate < 0:
                message = WheelsCmdStamped(vel_left=dtros.vel, vel_right=dtros.vel+abs(correctionUpdate))
            elif correctionUpdate > 0:
                message = WheelsCmdStamped(vel_left=dtros.vel+abs(correctionUpdate), vel_right=dtros.vel)
            else:
                message = WheelsCmdStamped(vel_left=dtros.vel, vel_right=dtros.vel)
            
            dtros._publisher.publish(message)
            
            rate.sleep()
        
        print("Crosswalk Found")
    
    def getCenterDistanceFromBottom(self, mask):
        """
        Computes the center of active pixel values in the mask and 
        the distance of the lowest active pixel from the bottom.
        Also checks if the active pixel count exceeds a given threshold.

        :param mask: Binary mask (numpy array) where active pixels are > 0
        :param threshold: Minimum number of active pixels required
        :return: distance_from_bottom if threshold is met, else math.inf
        """
        y_coords, x_coords = np.where(mask > 0)
        active_pixel_count = len(x_coords)

        if active_pixel_count < self.detection_magnitude:
            return math.inf  # Not enough active pixels

        # Compute distance from the bottom of the image
        img_height = mask.shape[0]
        distance_from_bottom = img_height - max(y_coords)

        return distance_from_bottom
    
            
class Stop(PDCrossWalkTask):
    def __init__(self, stop_time=3):
        super().__init__()
        self._stop_time = stop_time
    
    def runTask(self, dtros):
        print("Stopping")
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        dtros._publisher.publish(stop)
        time.sleep(self._stop_time)

class WaitForPeduckstrians(PDCrossWalkTask):
    def __init__(self, detection_magnitude=10000,  detection_variance=5000):
        super().__init__()

        self.detection_magnitude = detection_magnitude
        self.detection_variance = detection_variance

    def runTask(self, dtros):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            message = WheelsCmdStamped(vel_left=0, vel_right=0)
            dtros._publisher.publish(message)
            if dtros.homography_yellow_mask is not None and not self.isPeduckstriansVisible(dtros.homography_yellow_mask):
                break
            rate.sleep()
        print("No peduckstrians anymore")

    def isPeduckstriansVisible(self, mask):
        _, x_coords = np.where(mask > 0)

        if len(x_coords) == 0:
            return

        x_variance = np.var(x_coords)

        magnitude = len(x_coords)

        if (magnitude >= self.detection_magnitude):
            if(x_variance >= self.detection_variance):
                print("Peduckstrians detected")
                return True
        return False
    
class DrivePastCrossWalk(PDCrossWalkTask):
    def __init__(self, detection_magnitude):
        super().__init__()

        self.detection_magnitude = detection_magnitude

    def runTask(self, dtros):
        print("Driving past crosswalk")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            correctionUpdate = dtros.getUpdate()

            if dtros.homography_blue_mask is not None and not self.isDetectingCrossWalk(dtros.homography_blue_mask):
                break

            if correctionUpdate < 0:
                message = WheelsCmdStamped(vel_left=dtros.vel, vel_right=dtros.vel+abs(correctionUpdate))
            elif correctionUpdate > 0:
                message = WheelsCmdStamped(vel_left=dtros.vel+abs(correctionUpdate), vel_right=dtros.vel)
            else:
                message = WheelsCmdStamped(vel_left=dtros.vel, vel_right=dtros.vel)
            
            dtros._publisher.publish(message)
            
            rate.sleep()

    def isDetectingCrossWalk(self, homography_mask):
        """
        Checks if the homography mask exceeds a given threshold.

        :param homography_mask: Binary mask (numpy array) representing the homography
        :param threshold: Threshold for the active pixel count
        :return: True if the active pixel count exceeds the threshold, else False
        """
        # Count the number of active pixels in the mask
        active_pixel_count = np.count_nonzero(homography_mask > 0)

        if active_pixel_count > self.detection_magnitude:
            return True  # Crosswalk detected
        return False  # No crosswalk detected

    

if __name__ == '__main__':

    tasks = [
        FindCrossWalk(detection_magnitude=3000,  stopping_distance_in_pixels=5),
        WaitForPeduckstrians(detection_magnitude=2000,  detection_variance=5000),
        Stop(stop_time=1),
        DrivePastCrossWalk(detection_magnitude=3000),
        FindCrossWalk(detection_magnitude=3000,  stopping_distance_in_pixels=5),
        WaitForPeduckstrians(detection_magnitude=2000,  detection_variance=5000),
        Stop(stop_time=1),
        DrivePastCrossWalk(detection_magnitude=3000),
        FindCrossWalk(detection_magnitude=3000,  stopping_distance_in_pixels=5),
    ]
    # create the node
    node = PDCrossWalkMain(node_name="PDCrossWalkMain_node", tasks=tasks, loopTasks=False, proportional_gain=0.0000002, derivative_gain=0.0000002, integral_gain=0.0000002, velocity=0.3, integral_saturation=500000)
    node.run()
    # keep spinning
    rospy.spin()