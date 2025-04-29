#!/usr/bin/env python3
import dt_apriltags
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge
from enum import Enum
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import String
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA
from apriltags import AprilTag

class Colors(Enum):
    Red = [1.0, 0.0, 0.0]
    Green = [0.0, 1.0, 0.0]
    Blue = [0.0, 0.0, 1.0]
    Yellow = [1.0, 1.0, 0.0]
    Teal = [0.0, 1.0, 1.0]
    Magenta = [1.0, 0.0, 1.0]
    Off = [0.0, 0.0, 0.0]
    DarkOrange = [1.0, 0.55, 0]

class AprilTagDetection(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(AprilTagDetection, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

        # bridge between OpenCV and ROS
        self._bridge = CvBridge()

        self._state_topic = f"/{self._vehicle_name}/state"
        self._state_publisher = rospy.Publisher(self._state_topic, String, queue_size=1)

        self.apriltag_augmentation_topic = f"/{self._vehicle_name}/camera_node/apriltag_augmentation/compressed"
        self.apriltag_augmentation_publisher = rospy.Publisher(self.apriltag_augmentation_topic, CompressedImage)

        self.undistort_gray_topic = f"/{self._vehicle_name}/camera_node/undistort_gray/compressed"
        self.undistort_gray_sub = rospy.Subscriber(self.undistort_gray_topic, CompressedImage, self.undistorted_gray_callback)
        

        # Initialize AprilTag detector **only once**
        self.detector = dt_apriltags.Detector(families="tag36h11")

        self.detected_state = None

    def undistorted_gray_callback(self, msg):
        # Convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # If the image is grayscale (single channel), convert it to BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Detect AprilTags (assuming detector works on grayscale, you might need to convert back or detect on the original)
        results = self.detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        for r in results:
            # Extract the bounding box coordinates and convert to integers
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = (int(ptA[0]), int(ptA[1]))
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))

            # Draw the bounding box of the AprilTag detection in green
            cv2.line(image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(image, ptD, ptA, (0, 255, 0), 2)

            # Draw the center of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

            # Draw the tag id in green on the image
            cv2.putText(image, str(r.tag_id), (ptA[0], ptA[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish the augmented image
        self.apriltag_augmentation_publisher.publish(self._bridge.cv2_to_compressed_imgmsg(image))

        self.detected_state = "No tag detected"
        # Loop through detected tags and update detected_state
        for result in results:
            if result.tag_id == AprilTag.UALBERTA.value:
                self.detected_state = "UALBERTA tag detected"
            elif result.tag_id == AprilTag.INTERSECTIONT.value:
                self.detected_state =  "INTERSECTIONT tag detected"
            elif result.tag_id == AprilTag.STOP.value:
                self.detected_state = "STOP tag detected"
            break

        print(len(results))
        print(self.detected_state)
        self._state_publisher.publish(self.detected_state)

        


if __name__ == '__main__':
    # create the node
    node = AprilTagDetection(node_name='april_tag_detection_node')
    # node.run()
    # keep spinning
    rospy.spin()