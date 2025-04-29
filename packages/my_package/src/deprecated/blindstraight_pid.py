#!/usr/bin/env python3

import os
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
import cv2
from cv_bridge import CvBridge
import dt_apriltags

class PIDController(DTROS):

    def __init__(self, node_name, proportional_gain=0.0000002, derivative_gain=0.0000002, integral_gain=0.0000002, velocity=0.3, integral_saturation=500000):
        # initialize the DTROS parent class
        super(PIDController, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        
        self.proportional_gain = proportional_gain
        self.derivate_gain = derivative_gain
        self.integral_gain = integral_gain
        self.vel = velocity
        self.integral_saturation = integral_saturation

        self._error = 20
        self._error_last = self._error
        self._integration_stored = 0

        # Initialize AprilTag detector **only once**
        self.detector = dt_apriltags.Detector(families="tag36h11")

        self.detected_state = None

        self.apriltag_center_offset_error = 0

        self.yellow_line_verticality_error = 0

        self._ticks_left_cur = None
        self._ticks_right_cur = None

        self._ticks_left_start = None
        self._ticks_right_start = None

        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"

        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)


        # construct subscriber
        self._homography_two_lanes_wide_topic = f"/{self._vehicle_name}/camera_node/homography_two_lanes_wide/compressed"
        self.homography_two_lanes_wide_sub = rospy.Subscriber(self._homography_two_lanes_wide_topic, CompressedImage, self.callback_homography_two_lanes_wide)

        self._homography_white_mask_topic = f"/{self._vehicle_name}/camera_node/homography_white_mask/compressed"
        self.homography_white_mask_sub = rospy.Subscriber(self._homography_white_mask_topic, CompressedImage, self.callback_homography_white_mask)
    
    def callback_left(self, data):
        if self._ticks_left_start is None:
            self._ticks_left_start = data.data
            self._ticks_left_cur = data.data
            return
        
        self._ticks_left_cur = data.data
        self.updateError()
    
    def callback_right(self, data):
        if self._ticks_right_start is None:
            self._ticks_right_start = data.data
            self._ticks_right_cur = data.data
            return
        self._ticks_right_cur = data.data
        self.updateError()

    def updateError(self):
        if (self._ticks_right_cur is not None and 
        self._ticks_right_start is not None and 
        self._ticks_left_cur is not None and 
        self._ticks_left_start is not None):
            self._error = (self._ticks_right_cur - self._ticks_right_start) - (self._ticks_left_cur - self._ticks_left_start)
        print(self._error)

    def callback_homography_white_mask(self, msg):
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # self._error = self.compute_error(mask=image, target_x=50)

        # print(self._error)

    def callback_homography_two_lanes_wide(self, msg):
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # # Convert to grayscale
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # # Hough Transform to detect lines
        # lines = cv2.HoughLines(gray, 1, np.pi / 180, threshold=150)

        # if lines is not None:
        #     angles = []
        #     horiz_scores = []

        #     for line in lines:
        #         rho, theta = line[0]
        #         angle_deg = np.degrees(theta)
        #         horiz_score = min(abs(angle_deg), abs(angle_deg - 180))
        #         angles.append(angle_deg)
        #         horiz_scores.append(horiz_score)

        #     if len(angles) == 0 or len(horiz_scores) == 0:
        #         return

        #     avg_angle = sum(angles) / len(angles)
        #     avg_score = sum(horiz_scores) / len(horiz_scores)
        #     self._error = 90 - avg_angle
        #     print(f"Avg Angle: {avg_angle:.2f}°, Horizontality Score: {avg_score:.2f}")
        # # Show the result
        # cv2.imshow("Hough Lines", image)
        # cv2.waitKey(1)


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

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            correctionUpdate = self.getUpdate()
            
            # if correctionUpdate < 0:
            #     message = WheelsCmdStamped(vel_left=self.vel, vel_right=self.vel+abs(correctionUpdate))
            # elif correctionUpdate > 0:
            #     message = WheelsCmdStamped(vel_left=self.vel+abs(correctionUpdate), vel_right=self.vel)
            # else:
            message = WheelsCmdStamped(vel_left=self.vel, vel_right=self.vel)
            
            self._publisher.publish(message)
            
            rate.sleep()

    def getUpdate(self):
        P = self._error*self.proportional_gain
        errorRateOfChange = self._error - self._error_last
        D = self.derivate_gain * errorRateOfChange
        integration_stored_update = self._integration_stored + (self._error)
        self._integration_stored = (integration_stored_update) if abs(integration_stored_update) <= self.integral_saturation else (integration_stored_update/integration_stored_update)*self.integral_saturation
        I = self.integral_gain * self._integration_stored

        self._error_last = self._error

        return P + I + D
    
    def on_shutdown(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop)

if __name__ == '__main__':
    # create the node
    node = PIDController(node_name="PID_controller_node", proportional_gain=0.03, derivative_gain=0.03, integral_gain=0, velocity=0.3, integral_saturation=100)
    # node.run()
    # keep spinning
    rospy.spin()