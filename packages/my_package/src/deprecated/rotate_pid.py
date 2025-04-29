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

        self.white_line_error = 0

        self._homography_wide_white_mask_topic = f"/{self._vehicle_name}/camera_node/homography_wide_white_mask/compressed"
        self.homography_wide_white_mask_sub = rospy.Subscriber(self._homography_wide_white_mask_topic, CompressedImage, self.callback_homography_wide_white_mask)

    def callback_homography_wide_white_mask(self, msg):
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        height, width = image.shape[:2]

        edges = cv2.Canny(image, 50, 150)

        # Detect lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)

        if lines is not None:
            angles = []
            vertical_scores = []

            for line in lines:
                rho, theta = line[0]
                angle_deg = np.degrees(theta)
                angles.append(angle_deg)

            avg_angle = sum(angles) / len(angles)
            avg_verticality = 0 if 100 - min(abs(avg_angle), abs(180 - avg_angle)) <= 0 else 100 - min(abs(avg_angle), abs(180 - avg_angle))

            if avg_angle > 90:
                self._error = 100 - avg_verticality
            else:
                self._error = -1*(100 - avg_verticality)

            print(f"Average Angle: {avg_angle:.2f}°")
            print(f"Average Verticality Score: {avg_verticality:.2f}")
        
        cv2.imshow("Edges", edges)
        cv2.waitKey(1)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            correctionUpdate = self.getUpdate()
            
            if correctionUpdate < 0:
                message = WheelsCmdStamped(vel_left=self.vel+abs(correctionUpdate), vel_right=self.vel-abs(correctionUpdate))
            elif correctionUpdate > 0:
                message = WheelsCmdStamped(vel_left=self.vel-abs(correctionUpdate), vel_right=self.vel+abs(correctionUpdate))
            else:
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
    node = PIDController(node_name="PID_controller_node", proportional_gain=0.03, derivative_gain=0.03, integral_gain=0, velocity=0, integral_saturation=100)
    node.run()
    # keep spinning
    rospy.spin()