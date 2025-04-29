#!/usr/bin/env python3

import os
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
import cv2
from cv_bridge import CvBridge

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

        self._yellow_lines = []
        self._white_lines = []

        # construct subscriber
        self._undistorted_white_mask_topic = f"/{self._vehicle_name}/camera_node/undistorted_white_mask/compressed"
        self.undistorted_white_mask_sub = rospy.Subscriber(self._undistorted_white_mask_topic, CompressedImage, self.callback_undistorted_white_mask)

        # construct subscriber
        self._undistorted_yellow_mask_topic = f"/{self._vehicle_name}/camera_node/undistorted_yellow_mask/compressed"
        self.undistorted_yellow_mask_sub = rospy.Subscriber(self._undistorted_yellow_mask_topic, CompressedImage, self.callback_undistorted_yellow_mask)

    def callback_undistorted_white_mask(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        undistorted_mask_white = image
        height = undistorted_mask_white.shape[0]
        undistorted_mask_white[:int(height / 4), :] = 0

        lines_yellow = cv2.HoughLines(undistorted_mask_white, 1, np.pi / 180, threshold=300)
        
        lines = []

        if lines_yellow is not None:
            lines.extend(lines_yellow[:, 0])  # flatten from shape (N, 1, 2) to (N, 2)
        
        self._white_lines = lines

        self.updateError()
        
    def callback_undistorted_yellow_mask(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        undistorted_mask_white = image
        height = undistorted_mask_white.shape[0]
        undistorted_mask_white[:int(height / 4), :] = 0

        lines_yellow = cv2.HoughLines(undistorted_mask_white, 1, np.pi / 180, threshold=100)
        
        lines = []

        if lines_yellow is not None:
            lines.extend(lines_yellow[:, 0])  # flatten from shape (N, 1, 2) to (N, 2)

        self._yellow_lines = lines

        self.updateError()
            

    
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            correctionUpdate = self.getUpdate()
            
            if correctionUpdate < 0:
                message = WheelsCmdStamped(vel_left=self.vel, vel_right=self.vel+abs(correctionUpdate))
            elif correctionUpdate > 0:
                message = WheelsCmdStamped(vel_left=self.vel+abs(correctionUpdate), vel_right=self.vel)
            else:
                message = WheelsCmdStamped(vel_left=self.vel, vel_right=self.vel)
            
            self._publisher.publish(message)
            
            rate.sleep()

    def updateError(self):
        lines = []
        lines.extend(self._yellow_lines)  # flatten from shape (N, 1, 2) to (N, 2)
        lines.extend(self._white_lines)  # flatten from shape (N, 1, 2) to (N, 2)
        
        if lines:
            angles = []
            horiz_scores = []
            for rho, theta in lines:
                angle_deg = np.degrees(theta)
                horiz_score = min(abs(angle_deg), abs(angle_deg - 180))
                # print(f"Angle: {angle_deg:.2f}°, Horizontality Score: {horiz_score:.2f}")
                angles.append(angle_deg)
                horiz_scores.append(horiz_score)
            
            if len(angles) == 0 or horiz_scores == 0:
                return

            avg_angle = sum(angles) / len(angles)
            avg_score = sum(horiz_scores) / len(horiz_scores)

            if avg_angle > 90:
                self._error = 100 - (avg_score)
            elif avg_score < 90:
                self._error = -1 * (100 - (avg_score))
            else:
                self._error = 0
            # print(self._error)

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
    
    def on_shutdown(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop)

if __name__ == '__main__':
    # create the node
    node = PIDController(node_name="PID_controller_node", proportional_gain=0.01, derivative_gain=0.01, integral_gain=0, velocity=0.3, integral_saturation=100)
    node.run()
    # keep spinning
    rospy.spin()