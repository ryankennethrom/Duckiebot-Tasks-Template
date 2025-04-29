#!/usr/bin/env python3

import os
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String
import math
import time

class AprilTagBehaviorMain(DTROS):

    def __init__(self, node_name, main_task, proportional_gain=0.0000002, derivative_gain=0.0000002, integral_gain=0.0000002, velocity=0.3, integral_saturation=500000):
        # initialize the DTROS parent class
        super(AprilTagBehaviorMain, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # static parameters
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

        state_topic = f"/{self._vehicle_name}/state"
        self.sub = rospy.Subscriber(state_topic, String, self.state_callback)

        self.current_state = "No tag detected"

        self.red_mask_topic = f"/{self._vehicle_name}/camera_node/homography_red_mask/compressed"
        self.sub_red_mask = rospy.Subscriber(self.red_mask_topic, CompressedImage, self.callback_red_mask)

        self.red_mask = None

        self.main_task = main_task

    def callback_red_mask(self, data):
        image = self._bridge.compressed_imgmsg_to_cv2(data)
        self.red_count = self.count_active_pixels(image)
        self.red_mask = image
    
    def count_active_pixels(self, mask):
        # Ensure mask is in binary format (0 and 255 values)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Count the number of non-zero pixels
        active_pixels = cv2.countNonZero(binary_mask)
        
        return active_pixels

    def state_callback(self, msg):
        self.current_state = msg.data

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

        self._error = self.compute_error(mask=mask_yellow, target_x=100, pixel_value=1) + self.compute_error(mask=mask_white, target_x=489)
        # print(self._error)

        # # Display all images in separate windows
        # cv2.imshow("Before (Original Image)", image)  # Original image with source points
        # cv2.imshow("After (Warped Image)", warped)    # Warped image
        # # cv2.imshow("White Lane Detection", mask_white)  # White mask
        # # cv2.imshow("Yellow Lane Detection", mask_yellow)  # Yellow mask
        # cv2.waitKey(1)

    def run(self):
        while not rospy.is_shutdown():
            interrupt_task = self.main_task.execute(self)
            interrupt_task.execute(self)
        rospy.signal_shutdown(reason="tasks complete")
    
    def getUpdate(self):
        P = self._error*self.proportional_gain
        errorRateOfChange = self._error - self._error_last
        D = self.derivate_gain * errorRateOfChange
        integration_stored_update = self._integration_stored + (self._error)
        self._integration_stored = (integration_stored_update) if abs(integration_stored_update) <= self.integral_saturation else (integration_stored_update/integration_stored_update)*self.integral_saturation
        I = self.integral_gain * self._integration_stored

        self._error_last = self._error

        return P + I + D
    
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

    def getCenterDistanceFromBottom(self, mask, magnitude_threshold):
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

        if active_pixel_count < magnitude_threshold:
            return math.inf  # Not enough active pixels

        # Compute distance from the bottom of the image
        img_height = mask.shape[0]
        distance_from_bottom = img_height - max(y_coords)

        return distance_from_bottom
    
    def on_shutdown(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop)

class PID():
    def __init__(self, red_detection_magnitude=500):
         self.detection_magnitude=red_detection_magnitude

    def execute(self, dtros):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            correctionUpdate = dtros.getUpdate()

            if dtros.current_state == "STOP tag detected":
                return StopBehavior(red_detection_magnitude=self.detection_magnitude, stop_time=3)
            elif dtros.current_state == "UALBERTA tag detected":
                return StopBehavior(red_detection_magnitude=self.detection_magnitude, stop_time=1)
            elif dtros.current_state == "INTERSECTIONT tag detected":
                return StopBehavior(red_detection_magnitude=self.detection_magnitude, stop_time=2)
            elif dtros.red_mask is not None and dtros.getCenterDistanceFromBottom(dtros.red_mask, self.detection_magnitude) < 5:
                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._publisher.publish(message)
                rospy.sleep(0.5)
                return DrivePastRedLine(self.detection_magnitude)

            if correctionUpdate < 0:
                message = WheelsCmdStamped(vel_left=dtros.vel, vel_right=dtros.vel+abs(correctionUpdate))
            elif correctionUpdate > 0:
                message = WheelsCmdStamped(vel_left=dtros.vel+abs(correctionUpdate), vel_right=dtros.vel)
            else:
                message = WheelsCmdStamped(vel_left=dtros.vel, vel_right=dtros.vel)
            
            dtros._publisher.publish(message)
            
            rate.sleep()

class DrivePastRedLine():
    def __init__(self, red_detection_magnitude):
        self.detection_magnitude=red_detection_magnitude

    def execute(self, dtros):
        print("Driving past red line")
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            correctionUpdate = dtros.getUpdate()

            if dtros.red_mask is not None and dtros.getCenterDistanceFromBottom(dtros.red_mask, self.detection_magnitude) == math.inf:
                break

            if correctionUpdate < 0:
                message = WheelsCmdStamped(vel_left=dtros.vel, vel_right=dtros.vel+abs(correctionUpdate))
            elif correctionUpdate > 0:
                message = WheelsCmdStamped(vel_left=dtros.vel+abs(correctionUpdate), vel_right=dtros.vel)
            else:
                message = WheelsCmdStamped(vel_left=dtros.vel, vel_right=dtros.vel)
            
            dtros._publisher.publish(message)
            
            rate.sleep()

class StopBehavior():
    def __init__(self, red_detection_magnitude, stop_time=3):
        self.detection_magnitude=red_detection_magnitude
        self.stop_time = stop_time

    def execute(self, dtros):
        print("Finding red line to stop")
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            correctionUpdate = dtros.getUpdate()

            if dtros.getCenterDistanceFromBottom(dtros.red_mask, self.detection_magnitude) < 5:
                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._publisher.publish(message)
                rospy.sleep(self.stop_time)
                break

            if correctionUpdate < 0:
                message = WheelsCmdStamped(vel_left=dtros.vel, vel_right=dtros.vel+abs(correctionUpdate))
            elif correctionUpdate > 0:
                message = WheelsCmdStamped(vel_left=dtros.vel+abs(correctionUpdate), vel_right=dtros.vel)
            else:
                message = WheelsCmdStamped(vel_left=dtros.vel, vel_right=dtros.vel)
            
            dtros._publisher.publish(message)
            
            rate.sleep()

if __name__ == '__main__':
    # create the node
    node = AprilTagBehaviorMain(node_name="PID_controller_node", main_task=PID(),proportional_gain=0.0000002, derivative_gain=0.0000002, integral_gain=0.0000002, velocity=0.3, integral_saturation=500000)
    node.run()
    # keep spinning
    rospy.spin()