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

        # self._homography_white_mask_topic = f"/{self._vehicle_name}/camera_node/homography_white_mask/compressed"
        # self.homography_white_mask_sub = rospy.Subscriber(self._homography_white_mask_topic, CompressedImage, self.callback_homography_white_mask)

        self._homography_wide_white_mask_topic = f"/{self._vehicle_name}/camera_node/homography_wide_white_mask/compressed"
        self.homography_wide_white_mask_sub = rospy.Subscriber(self._homography_wide_white_mask_topic, CompressedImage, self.callback_homography_wide_white_mask)

        self._undistort_gray_topic = f"/{self._vehicle_name}/camera_node/undistort_gray/compressed"
        self.undistort_gray_sub = rospy.Subscriber(self._undistort_gray_topic, CompressedImage, self.callback_undistort_gray)

    def callback_undistort_gray(self, msg):
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # # If the image is grayscale (single channel), convert it to BGR
        # if len(image.shape) == 2:
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # # Detect AprilTags (assuming detector works on grayscale, you might need to convert back or detect on the original)
        # results = self.detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        # for r in results:
        #     # Extract the bounding box coordinates and convert to integers
        #     (ptA, ptB, ptC, ptD) = r.corners
        #     ptA = (int(ptA[0]), int(ptA[1]))
        #     ptB = (int(ptB[0]), int(ptB[1]))
        #     ptC = (int(ptC[0]), int(ptC[1]))
        #     ptD = (int(ptD[0]), int(ptD[1]))

        #     # Draw the bounding box of the AprilTag detection in green
        #     cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        #     cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        #     cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        #     cv2.line(image, ptD, ptA, (0, 255, 0), 2)

        #     # Draw the center of the AprilTag
        #     (cX, cY) = (int(r.center[0]), int(r.center[1]))
        #     cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

        #     # Draw the tag id in green on the image
        #     cv2.putText(image, str(r.tag_id), (ptA[0], ptA[1] - 15),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #     if r.tag_id == 13:
        #         print("tag id 13 found")
        #         image_center_x = image.shape[1] // 2  # width // 2
        #         tag_center_x = int(r.center[0])
        #         self.apriltag_center_offset_error = tag_center_x - image_center_x

        # cv2.imshow("AprilTag Detection", image)
        # cv2.waitKey(1)


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

                # Verticality score: 0° or 180° means perfectly vertical, 90° is horizontal
                vertical_score = 0 if 100 - min(abs(angle_deg), abs(180 - angle_deg)) <= 0 else 100 - min(abs(angle_deg), abs(180 - angle_deg))

                angles.append(angle_deg)
                vertical_scores.append(vertical_score)

            avg_angle = sum(angles) / len(angles)
            avg_verticality = sum(vertical_scores) / len(vertical_scores)

            if avg_angle > 90:
                self._error = 100 - avg_verticality
            else:
                self._error = -1*(100 - avg_verticality)
            print(f"Average Angle: {avg_angle:.2f}°")
            print(f"Average Verticality Score: {avg_verticality:.2f}")
        else:
            print("No lines detected.")
        
        cv2.imshow("Edges", image)
        cv2.waitKey(1)


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
    node = PIDController(node_name="PID_controller_node", proportional_gain=0.05, derivative_gain=0.05, integral_gain=0, velocity=-0.3, integral_saturation=100)
    node.run()
    # keep spinning
    rospy.spin()