from custom_utils.dtros_operations import DtrosOperations
from std_msgs.msg import String
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped, LEDPattern
import math
import time
from std_msgs.msg import ColorRGBA, Header
import types
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from custom_utils.mask_operations import MaskOperations
from custom_utils.pid_operations import PIDOperations
from custom_utils.image_operations import ImageOperations
from custom_utils.tag_r_operations import TagROperations
import dt_apriltags
import numpy as np
import rospy
import os
from custom_utils.constants import Stall, Tag
from custom_utils.color_operations import *

class FinalBehaviorMainTask():
    def execute(self, dtros):
        print(str(self.__class__.__name__) + " started")
        self.onStart(dtros)
        self.runTask(dtros)
        self.onTearDown(dtros)

    def onStart(self, dtros):
        raise Exception("onStart() must be overriden")
    
    def onTearDown(self, dtros):
        DtrosOperations.unregister_and_delete_subscribers(self)
        print(str(self.__class__.__name__) + " exited")
        
    def runTask(self, dtros):
        raise Exception("runTask() must be overriden")
    
class RawImageTask(FinalBehaviorMainTask):
    def __init__(self):
        self._raw_image = None
        self._bridge = CvBridge()

    def onStart(self, dtros):
        vehicle_name = os.environ["VEHICLE_NAME"]
        raw_image_topic = f"/{vehicle_name}/camera_node/image/compressed"
        self._sub_raw_image = rospy.Subscriber(raw_image_topic, CompressedImage, self.callback_raw_image)

    def callback_raw_image(self, msg):
        self._raw_image = self._bridge.compressed_imgmsg_to_cv2(msg)

    def runTask(self, dtros):
        return super().runTask(dtros)
    
class HomographyTask(RawImageTask):
    def __init__(self):
        super().__init__()
        self._undistort = None
        self._homography = None

    def callback_raw_image(self, msg):
        super().callback_raw_image(msg)
        self._undistort = ImageOperations.undistort(self._raw_image) 
        self._homography = ImageOperations.getHomography(self._undistort)

    def runTask(self, dtros):
        return super().runTask(dtros)
    
class TurnRightTask(FinalBehaviorMainTask):
    #R = 0.35
    def __init__(self, precision=40, tolerance=0, radians=math.pi/5, angular_velocity=1.5, R=0.2):
        super().__init__()
        self._precision = precision
        self._tolerance = tolerance
        self._radians = radians
        self._angular_velocity=angular_velocity
        self._R = R

        self._ticks_left = None
        self._ticks_right = None

        self._distance_left = 0
        self._distance_right = 0

        self._resolution = 135
        self._l = 0.050

    def onStart(self, dtros):

        vehicle_name = os.environ["VEHICLE_NAME"]
        wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"

        self._radius = rospy.get_param(f'/{vehicle_name}/kinematics_node/radius', 0.0318)
        
        left_encoder_topic = f"/{vehicle_name}/left_wheel_encoder_node/tick"
        right_encoder_topic = f"/{vehicle_name}/right_wheel_encoder_node/tick"

        self._sub_left_wheel = rospy.Subscriber(left_encoder_topic, WheelEncoderStamped, self.callback_left_wheel)
        self._sub_right_wheel = rospy.Subscriber(right_encoder_topic, WheelEncoderStamped, self.callback_right_wheel)
        self._wheels_publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

    def callback_left_wheel(self, data):
        if self._ticks_left is None:
            self._ticks_left = data.data
            return
        
        self._distance_left += 2*math.pi*self._radius*((data.data - self._ticks_left)/self._resolution)
        self._ticks_left = data.data
    
    def callback_right_wheel(self, data):
        if self._ticks_right is None:
            self._ticks_right = data.data
            return
        
        self._distance_right += 2*math.pi*self._radius*((data.data - self._ticks_right)/self._resolution)
        self._ticks_right = data.data

    def runTask(self, dtros):
        
        precision = self._precision
        tolerance = self._tolerance
        target_radian = self._radians
        angular_velocity = self._angular_velocity
        R = self._R

        # msg = f""" Running a curve task ... target_angle : {target_radian}, R: {R}, precision: {precision}, tolerance: {tolerance} """
        # rospy.loginfo(msg)

        rate = rospy.Rate(precision)

        v_r = (R - self._l*3) * angular_velocity
        v_l = (R + self._l*3) * angular_velocity

        message = WheelsCmdStamped(vel_left=v_l, vel_right=v_r)

        while not rospy.is_shutdown():
            total_change_angle = ( self._distance_right - self._distance_left ) / (2*self._l)
            # msg = f""" total_change_angle: {total_change_angle}, target_radian {target_radian}"""
            # rospy.loginfo(msg)
            if abs(total_change_angle) >= (abs(target_radian) - tolerance):
                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._wheels_publisher.publish(message)
                break

            dtros._wheels_publisher.publish(message)
            rate.sleep()

        # msg = f""" total_change_angle: {total_change_angle}, target_radian {target_radian}"""
        # rospy.loginfo(msg)

class PulsingRightTurnTask(TurnRightTask):
    def __init__(self, precision=40, tolerance=0, radians=math.pi/5, angular_velocity=1.5, R=0.2,
                 pulse_duration=0.2, pause_duration=0.3):
        super().__init__(precision, tolerance, radians, angular_velocity, R)
        self._pulse_duration = pulse_duration
        self._pause_duration = pause_duration

    def runTask(self, dtros):
        precision = self._precision
        tolerance = self._tolerance
        target_radian = self._radians
        angular_velocity = self._angular_velocity
        R = self._R
        pulse_duration = self._pulse_duration
        pause_duration = self._pause_duration

        self._distance_left = 0
        self._distance_right = 0
        self._ticks_left = None
        self._ticks_right = None

        v_r = (R - self._l * 3) * angular_velocity
        v_l = (R + self._l * 3) * angular_velocity

        message = WheelsCmdStamped(vel_left=v_l, vel_right=v_r)
        stop_msg = WheelsCmdStamped(vel_left=0, vel_right=0)

        pulsing = True
        last_pulse_time = time.time()

        rate = rospy.Rate(precision)

        while not rospy.is_shutdown():
            current_time = time.time()
            total_change_angle = (self._distance_right - self._distance_left) / (2 * self._l)

            if abs(total_change_angle) >= (abs(target_radian) - tolerance):
                dtros._wheels_publisher.publish(stop_msg)
                break

            # Send pulse or pause
            if pulsing:
                vel = 0.2
                message = WheelsCmdStamped(vel_left=vel, vel_right=-vel)
                dtros._wheels_publisher.publish(message)
            else:
                dtros._wheels_publisher.publish(WheelsCmdStamped(vel_left=0, vel_right=0))

            # Toggle pulse/pause state
            if pulsing and (current_time - last_pulse_time) > pulse_duration:
                pulsing = False
                last_pulse_time = current_time
            elif not pulsing and (current_time - last_pulse_time) > pause_duration:
                pulsing = True
                last_pulse_time = current_time

            rate.sleep()

class TurnLeftTask(FinalBehaviorMainTask):
    # R = 0.4445
    def __init__(self, precision=40, tolerance=0.04, radians=math.pi/2, angular_velocity=2, R=0.45):
        super().__init__()
        self._precision = precision
        self._tolerance = tolerance
        self._radians = radians
        self._angular_velocity=angular_velocity
        self._R = R

        self._ticks_left = None
        self._ticks_right = None

        self._distance_left = 0
        self._distance_right = 0

        self._resolution = 135
        self._l = 0.050

    def onStart(self, dtros):

        vehicle_name = os.environ["VEHICLE_NAME"]
        wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"

        self._radius = rospy.get_param(f'/{vehicle_name}/kinematics_node/radius', 0.0318)
        
        left_encoder_topic = f"/{vehicle_name}/left_wheel_encoder_node/tick"
        right_encoder_topic = f"/{vehicle_name}/right_wheel_encoder_node/tick"

        self._sub_left_wheel = rospy.Subscriber(left_encoder_topic, WheelEncoderStamped, self.callback_left_wheel)
        self._sub_right_wheel = rospy.Subscriber(right_encoder_topic, WheelEncoderStamped, self.callback_right_wheel)
        self._wheels_publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

    def callback_left_wheel(self, data):
        if self._ticks_left is None:
            self._ticks_left = data.data
            return
        
        self._distance_left += 2*math.pi*self._radius*((data.data - self._ticks_left)/self._resolution)
        self._ticks_left = data.data
    
    def callback_right_wheel(self, data):
        if self._ticks_right is None:
            self._ticks_right = data.data
            return
        
        self._distance_right += 2*math.pi*self._radius*((data.data - self._ticks_right)/self._resolution)
        self._ticks_right = data.data

    def runTask(self, dtros):
        
        precision = self._precision
        tolerance = self._tolerance
        target_radian = self._radians
        angular_velocity = self._angular_velocity
        R = self._R

        # msg = f""" Running a curve task ... target_angle : {target_radian}, R: {R}, precision: {precision}, tolerance: {tolerance} """
        # rospy.loginfo(msg)

        rate = rospy.Rate(precision)

        v_l = (R - self._l*3) * angular_velocity
        v_r = (R + self._l*3) * angular_velocity

        message = WheelsCmdStamped(vel_left=v_l, vel_right=v_r)

        while not rospy.is_shutdown():
            total_change_angle = ( self._distance_right - self._distance_left ) / (2*self._l)
            # msg = f""" total_change_angle: {total_change_angle}, target_radian {target_radian}"""
            # rospy.loginfo(msg)
            if abs(total_change_angle) >= (abs(target_radian) - tolerance):
                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._wheels_publisher.publish(message)
                break

            dtros._wheels_publisher.publish(message)
            rate.sleep()

        # msg = f""" total_change_angle: {total_change_angle}, target_radian {target_radian}"""
        # rospy.loginfo(msg)

class PulsingLeftTurnTask(TurnLeftTask):
    def __init__(self, precision=40, tolerance=0.04, radians=math.pi/2, angular_velocity=2, R=0.45,
                 pulse_duration=0.2, pause_duration=0.3):
        super().__init__(precision, tolerance, radians, angular_velocity, R)
        self._pulse_duration = pulse_duration
        self._pause_duration = pause_duration

    def runTask(self, dtros):
        precision = self._precision
        tolerance = self._tolerance
        target_radian = self._radians
        angular_velocity = self._angular_velocity
        R = self._R
        pulse_duration = self._pulse_duration
        pause_duration = self._pause_duration

        self._distance_left = 0
        self._distance_right = 0
        self._ticks_left = None
        self._ticks_right = None

        v_l = (R - self._l * 3) * angular_velocity
        v_r = (R + self._l * 3) * angular_velocity

        stop_msg = WheelsCmdStamped(vel_left=0, vel_right=0)

        pulsing = True
        last_pulse_time = time.time()

        rate = rospy.Rate(precision)

        while not rospy.is_shutdown():
            current_time = time.time()
            total_change_angle = (self._distance_right - self._distance_left) / (2 * self._l)

            if abs(total_change_angle) >= (abs(target_radian) - tolerance):
                dtros._wheels_publisher.publish(stop_msg)
                break

            # Send pulse or pause
            if pulsing:
                vel = 0.2
                message = WheelsCmdStamped(vel_left=-vel, vel_right=vel)
                dtros._wheels_publisher.publish(message)
            else:
                dtros._wheels_publisher.publish(WheelsCmdStamped(vel_left=0, vel_right=0))

            if pulsing and (current_time - last_pulse_time) > pulse_duration:
                pulsing = False
                last_pulse_time = current_time
            elif not pulsing and (current_time - last_pulse_time) > pause_duration:
                pulsing = True
                last_pulse_time = current_time

            rate.sleep()

class StallAlignmentTask(FinalBehaviorMainTask):
    def __init__(self, target_stall, proportional_gain, derivative_gain, integral_gain, velocity, integral_saturation):
        
        if not isinstance(target_stall, Stall):
            raise Exception("stall must be of type Stall(Enum)")

        super().__init__()
        
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

        self._homography_white_mask = None

        self._target_stall = target_stall

        self._curr_drive_direction = "Forward"

    def callback_raw_image(self, msg):
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        undistorted = ImageOperations.undistort(image)
        homography = ImageOperations.getHomography(undistorted)
        self._homography_white_mask = ImageOperations.getWhiteMask(homography)
        yellow_mask = ImageOperations.getYellowMask(homography)
        white_mask = self._homography_white_mask

        lines_white = ImageOperations.getMaskLines(white_mask)

        lines_yellow = ImageOperations.getMaskLines(yellow_mask)

        self._white_lines = lines_white
        self._yellow_lines = lines_yellow

        self.updateError()

        # cv2.imshow("Undistort White Mask", image)
        # cv2.waitKey(1)

    def onStart(self, dtros):

        # construct subscriber
        self._raw_image_topic = f"/{dtros._vehicle_name}/camera_node/image/compressed"
        self._sub_raw_image = rospy.Subscriber(self._raw_image_topic, CompressedImage, self.callback_raw_image)

    def runTask(self, dtros):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            correctionUpdate = self.getUpdate()

            if self._curr_drive_direction == "Forward" and MaskOperations.getActiveCount(self._homography_white_mask) > 10000 and MaskOperations.getActiveCenter(self._homography_white_mask)[1] > 300:
                if self._target_stall in [Stall.TWO, Stall.FOUR]:
                    break
                self._curr_drive_direction = "Backward"

            if self._curr_drive_direction == "Backward" and MaskOperations.getActiveCenter(self._homography_white_mask)[1] < 150:
                break

            if self._curr_drive_direction == "Forward":
                if correctionUpdate < 0:
                    message = WheelsCmdStamped(vel_left=self.vel, vel_right=self.vel+abs(correctionUpdate))
                elif correctionUpdate > 0:
                    message = WheelsCmdStamped(vel_left=self.vel+abs(correctionUpdate), vel_right=self.vel)
                else:
                    message = WheelsCmdStamped(vel_left=self.vel, vel_right=self.vel)
            else:
                if correctionUpdate < 0:
                    message = WheelsCmdStamped(vel_left=-1*self.vel-abs(correctionUpdate), vel_right=-1*self.vel)
                elif correctionUpdate > 0:
                    message = WheelsCmdStamped(vel_left=-1*self.vel, vel_right=-1*self.vel-abs(correctionUpdate))
                else:
                    message = WheelsCmdStamped(vel_left=-1*self.vel, vel_right=-1*self.vel)
            
            dtros._wheels_publisher.publish(message)
            
            rate.sleep()

        message = WheelsCmdStamped(vel_left=0, vel_right=0)
        dtros._wheels_publisher.publish(message)

        if self._target_stall in [Stall.ONE, Stall.TWO]:
            rospy.loginfo("Turning Right TASK")
            PulsingRightTurnTask(radians=math.pi/2).execute(dtros)
        else:
            rospy.loginfo("Turning Right TASK")
            PulsingLeftTurnTask(radians=math.pi/2).execute(dtros)

    def updateError(self):
        lines = []
        lines.extend(self._yellow_lines) 
        lines.extend(self._white_lines)
        if lines:
            angles = []
            horiz_scores = []
            for rho, theta in lines:
                angle_deg = np.degrees(theta)
                angles.append(angle_deg)

            if len(angles) == 0 or horiz_scores == 0:
                return

            avg_angle = sum(angles) / len(angles)
            avg_score = 0 if 100 - abs(90 - avg_angle) <= 0 else 100 - abs(90 - avg_angle)

            # print(f"Average Angle: {avg_angle:.2f}°, Average Score: {avg_score:.2f}")

            if avg_angle > 90:
                self._error = 100 - (avg_score)
            elif avg_score < 90:
                self._error = -1 * (100 - (avg_score))
            else:
                self._error = 0
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
    
class ForwardParkingTask(FinalBehaviorMainTask):
    def __init__(self, target_stall):
        self._target_stall = target_stall
        self.detector = dt_apriltags.Detector(families="tag36h11")
        self._bridge = CvBridge()
        self._target_tag_error = 0
        self._white_line_error = 0

        self._error_last = 0

        self._tag_perimeter = 0
        self._debug = False

    def onStart(self, dtros):
        vehicle_name = os.environ["VEHICLE_NAME"]
        self._raw_image_topic = f"/{vehicle_name}/camera_node/image/compressed"
        self._sub_raw_image = rospy.Subscriber(self._raw_image_topic, CompressedImage, self.callback_raw_image)

    def callback_raw_image(self, msg):
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        undistort = ImageOperations.undistort(image)
        undistort_gray = ImageOperations.getGrayscale(undistort)

        # Convert JPEG bytes to CV image
        image = undistort_gray

        height, width = image.shape[:2]

        # If the image is grayscale (single channel), convert it to BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Detect AprilTags (assuming detector works on grayscale, you might need to convert back or detect on the original)
        results = self.detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        for r in results:
            if r.tag_id == self._target_stall.value:
            # Extract the bounding box coordinates and convert to integers
                (ptA, ptB, ptC, ptD) = r.corners
                ptA = (int(ptA[0]), int(ptA[1]))
                ptB = (int(ptB[0]), int(ptB[1]))
                ptC = (int(ptC[0]), int(ptC[1]))
                ptD = (int(ptD[0]), int(ptD[1]))

                self._tag_perimeter = (
                    np.linalg.norm(np.array(ptA) - np.array(ptB)) +
                    np.linalg.norm(np.array(ptB) - np.array(ptC)) +
                    np.linalg.norm(np.array(ptC) - np.array(ptD)) +
                    np.linalg.norm(np.array(ptD) - np.array(ptA))
                )

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
                
                self._target_tag_error = (r.center[0] - (width // 2))*0.005
                if self._debug == True:
                    print(self._target_tag_error)
                    # cv2.imshow("Detection", image)
                    # cv2.waitKey(1)
                return
            
        self._target_tag_error = 0
        
        # cv2.imshow("Detection", image)
        # cv2.waitKey(1)

    def runTask(self, dtros):
        rate = rospy.Rate(10)  # 10 Hz
        pulse_duration = 0.2   # seconds of active movement
        pause_duration = 0.3   # seconds of no movement
        last_pulse_time = 0
        pulsing = True

        while not rospy.is_shutdown():
            current_time = time.time()

            if self._tag_perimeter >= 500:
                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._wheels_publisher.publish(message)
                break

            # Compute PID message for forward movement
            _, error_last_updated, message = PIDOperations.getForwardPIDWheelMsg(
                base_velocity=0.3,
                error_last=self._error_last,
                integration_stored=0,
                error=self._target_tag_error,
                proportional_gain=1,
                derivative_gain=1,
                integral_gain=0,
                integral_saturation=0
            )
            self._error_last = error_last_updated

            # Send pulse or pause
            if pulsing:
                self._error = self._target_tag_error
                vel = 0.6
                if self._error > 0:
                    message = WheelsCmdStamped(vel_left=vel, vel_right=0)
                elif self._error < 0:
                    message = WheelsCmdStamped(vel_left=0, vel_right=vel)
                else:
                    message = WheelsCmdStamped(vel_left=vel, vel_right=vel)
                dtros._wheels_publisher.publish(message)
            else:
                dtros._wheels_publisher.publish(WheelsCmdStamped(vel_left=0, vel_right=0))

            # Pulse timing logic
            if pulsing and (current_time - last_pulse_time) > pulse_duration:
                pulsing = False
                last_pulse_time = current_time
            elif not pulsing and (current_time - last_pulse_time) > pause_duration:
                pulsing = True
                last_pulse_time = current_time

            rate.sleep()

class LaneFollowing(FinalBehaviorMainTask):
    # integral_gain=0.0000002
    def __init__(self, base_velocity=0.3, integral_gain=0, debug=False):
        self._bridge = CvBridge()
        self._error_last = 0
        self._error = 0
        self._integration_stored = 0
        self._base_velocity = base_velocity
        self._homography = None
        self._undistorted = None
        self._mask_white = None
        self._mask_yellow = None
        self._debug = debug
        self._raw_image = None
        self._integral_gain = integral_gain

    def onStart(self, dtros):
        vehicle_name = os.environ["VEHICLE_NAME"]
        self._raw_image_topic = f"/{vehicle_name}/camera_node/image/compressed"
        self._sub_raw_image = rospy.Subscriber(self._raw_image_topic, CompressedImage, self.callback_raw_image)

    def getError(self):
        yellow_error = MaskOperations.computeErrorInAxisX(mask=self._mask_yellow, target_x=100, pixel_value=1.5) # original was 1.5
        white_error = MaskOperations.computeErrorInAxisX(mask=self._mask_white, target_x=489, pixel_value=1) # original was 489

        return yellow_error + white_error

    def callback_raw_image(self, msg):
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        self._raw_image = image
        self._undistorted = ImageOperations.undistort(image)
        self._homography = ImageOperations.getHomography(self._undistorted)

        self._mask_white = ImageOperations.getWhiteMask(self._homography)
        self._mask_yellow = ImageOperations.getYellowMask(self._homography)
        
        self._error = self.getError()

        # if self._debug == True:
        #     cv2.imshow("White Mask", self._mask_white)
        #     cv2.imshow("Raw Image", image)
        #     cv2.imshow("Yellow Mask", self._mask_yellow)
        #     cv2.waitKey(1)
        #     self._undistorted_white_mask_publisher.publish(self._bridge.cv2_to_compressed_imgmsg(self._mask_white))
        #     self._undistorted_yellow_mask_publisher.publish(self._bridge.cv2_to_compressed_imgmsg(self._mask_yellow))
    
    def isTimeToStop(self):
        return False
    
    def runTask(self, dtros):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.isTimeToStop():
                break
            integration_stored_updated, error_last_updated, message = PIDOperations.getForwardPIDWheelMsg(base_velocity=self._base_velocity, error_last=self._error_last, integration_stored=self._integration_stored, error=self._error, proportional_gain=0.0000002, derivative_gain=0.0000002, integral_gain=self._integral_gain, integral_saturation=500000)
            self._error_last = error_last_updated
            self._integration_stored = integration_stored_updated
            dtros._wheels_publisher.publish(message)
            rate.sleep()

class FindBrokenBot(LaneFollowing):
    def __init__(self, detection_threshold):
        super().__init__()
        self._duckiebot_detected_and_ready_to_switch = False
        self._detection_threshold = detection_threshold

    def callback_raw_image(self, msg):
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        undistorted = ImageOperations.undistort(image)
        homography = ImageOperations.getHomography(undistorted)
        mask_white = ImageOperations.getWhiteMask(homography)
        mask_yellow = ImageOperations.getYellowMask(homography)
        mask_blue = ImageOperations.getDuckiebotBlueMask(undistorted)

        yellow_error = MaskOperations.computeErrorInAxisX(mask=mask_yellow, target_x=489, pixel_value=1)
        white_error = MaskOperations.computeErrorInAxisX(mask=mask_white, target_x=100, pixel_value=1)

        mask_blue = MaskOperations.keep_middle_half(mask_blue)

        if MaskOperations.getActiveCount(mask_blue) > self._detection_threshold and yellow_error < -100000 and white_error > 100000:
            self._duckiebot_detected_and_ready_to_switch = True
        
        yellow_error = MaskOperations.computeErrorInAxisX(mask=mask_yellow, target_x=100, pixel_value=1)
        white_error = MaskOperations.computeErrorInAxisX(mask=mask_white, target_x=489, pixel_value=1)

        self._error = yellow_error + white_error

    def isTimeToStop(self):
        return self._duckiebot_detected_and_ready_to_switch


class SwitchLanesUntilSafe(LaneFollowing):
    def __init__(self, detection_threshold, safe_timer=2, base_velocity=0.3):
        super().__init__(base_velocity=base_velocity)
        self._duckie_detected_time_stamp = None
        self._detection_threshold = detection_threshold
        self._safe_timer = safe_timer

    def callback_raw_image(self, msg):
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        undistorted = ImageOperations.undistort(image)
        homography = ImageOperations.getHomography(undistorted)
        mask_white = ImageOperations.getWhiteMask(homography)
        mask_yellow = ImageOperations.getYellowMask(homography)

        mask_blue = ImageOperations.getDuckiebotBlueMask(undistorted)

        if MaskOperations.getActiveCount(mask_blue) > self._detection_threshold:
            self._duckie_detected_time_stamp = time.time()

        mask_yellow[:, :90] = 0
        yellow_error = MaskOperations.computeErrorInAxisX(mask=mask_yellow, target_x=489, pixel_value=1)
        white_error = MaskOperations.computeErrorInAxisX(mask=mask_white, target_x=100, pixel_value=1)

        if yellow_error > 100000 and white_error < -100000:
            self._error = (yellow_error + abs(white_error)) 
        elif yellow_error < -100000 and white_error > 100000:
            self._error = (yellow_error + -1*white_error) 
        else:
            self._error = yellow_error + white_error
    
    def isTimeToStop(self):
        if self._duckie_detected_time_stamp is not None and (time.time() - self._duckie_detected_time_stamp) > self._safe_timer:
            # print("Did not see a duckiebot for "+ str(self._safe_timer) + " seconds !")
            return True
        return False
    
class RightLaneFollowingWithLaneCorrection(LaneFollowing):
    def __init__(self, base_velocity=0.25, debug=True):
        super().__init__(base_velocity=base_velocity, debug=debug)

    def getError(self):
        self._mask_yellow[:, :90] = 0
        yellow_error = MaskOperations.computeErrorInAxisX(mask=self._mask_yellow, target_x=100, pixel_value=1.5)
        white_error = MaskOperations.computeErrorInAxisX(mask=self._mask_white, target_x=489, pixel_value=1)

        if yellow_error > 100000 and white_error < -100000:
            out = (yellow_error + abs(white_error)) 
        elif yellow_error < -100000 and white_error > 100000:
            out = (yellow_error + -1*white_error) 
        else:
            out = yellow_error + white_error
        return out
    
class RightLaneUntilCrosswalk(RightLaneFollowingWithLaneCorrection):
    def __init__(self, base_velocity=0.25, debug=False):
        super().__init__(base_velocity=base_velocity, debug=debug)

        self._crosswalk_detected_and_close_enough = False
    
    def callback_raw_image(self, msg):
        super().callback_raw_image(msg)
        cross_walk_blue_mask = ImageOperations.getCrosswalkBlueMask(self._homography)

        # MaskOperations.getActiveCenterX(cross_walk_blue_mask) > (3*ImageOperations.getCenterAxisY(cross_walk_blue_mask))//4
        if MaskOperations.getActiveCount(cross_walk_blue_mask) > 1000:
            self._crosswalk_detected_and_close_enough = True
        else:
            self._crosswalk_detected_and_close_enough = False

        # if self._debug == True:
        #     # print(MaskOperations.getActiveCount(cross_walk_blue_mask))
        #     # print(self._crosswalk_detected_and_close_enough)
        #     cv2.imshow("Crosswalk Blue Mask", cross_walk_blue_mask)
        #     cv2.waitKey(1)

    def isTimeToStop(self):
        return self._crosswalk_detected_and_close_enough

class WhiteLaneUntilCrossWalk(LaneFollowing):
    def __init__(self, debug=False):
        super().__init__()
        self._crosswalk_detected_and_close_enough = False
        self._debug = debug

    def callback_raw_image(self, msg):
        super().callback_raw_image(msg)
        cross_walk_blue_mask = ImageOperations.getCrosswalkBlueMask(self._homography)
        if MaskOperations.getActiveCount(cross_walk_blue_mask) > 1500 and MaskOperations.getLowestY(cross_walk_blue_mask) > 300:
            self._crosswalk_detected_and_close_enough = True
        else:
            self._crosswalk_detected_and_close_enough = False
        # if self._debug == True:
        #     cv2.imshow("Crosswalk Blue Mask", cross_walk_blue_mask)
        #     cv2.waitKey(1)
    

    def getError(self):
        white_error = MaskOperations.computeErrorInAxisX(mask=self._mask_white, target_x=489, pixel_value=1)

        return white_error

    def isTimeToStop(self):
        return self._crosswalk_detected_and_close_enough
    
    
class FreezeUntilDucksPass(HomographyTask):
    def __init__(self, unfreeze_delay=3):
        super().__init__()
        self._ducks_passing_time_stamp = None
        self._unfreeze_delay = unfreeze_delay
        self._yellow_mask = None
        self._ducks_passing = True

    def callback_raw_image(self, msg):
        super().callback_raw_image(msg)
        self._yellow_mask = ImageOperations.getYellowMask(self._homography)
        self._ducks_passing = self.areDucksPassing(self._yellow_mask)
    
    def areDucksPassing(self, yellow_mask):
        if MaskOperations.getActiveCount(yellow_mask) >= 5000 and MaskOperations.getMaskVarianceAxisX(yellow_mask) >= 5000:
            return True
        return False
    
    def runTask(self, dtros):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if not self._ducks_passing:
                break
            message = WheelsCmdStamped(vel_left=0, vel_right=0)
            dtros._wheels_publisher.publish(message)
            rate.sleep()

class LaneFollowUntilIntersection(LaneFollowing):
    def __init__(self, base_velocity, debug=False):
        super().__init__(base_velocity=base_velocity, debug=debug)
        self._red_mask = None

    def callback_raw_image(self, msg):
        super().callback_raw_image(msg)
        self._red_mask = ImageOperations.getRedMask(self._homography)

        # if self._debug == True:
        #     cv2.imshow("Homography Red Mask", self._red_mask)
        #     cv2.waitKey(1)
    
    def isTimeToStop(self):
        if self._debug == True:
            # print(MaskOperations.getActiveCenter(self._red_mask)[1])
            # print(MaskOperations.getMaskVarianceAxisX(self._red_mask)) if self._red_mask is not None else None
            #return False
            pass
        if self._red_mask is not None and MaskOperations.getActiveCount(self._red_mask) > 1000 and MaskOperations.getActiveCenter(self._red_mask)[1] > 300:
            return True
        else:
            return False
    

class DriveOverRedline(LaneFollowUntilIntersection):
    def __init__(self, base_velocity=0.5, red_count_target=10000, debug=True):
        super().__init__(base_velocity=base_velocity, debug=debug)
        self._red_count_target = red_count_target

    def runTask(self, dtros):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self._red_mask is not None and MaskOperations.getActiveCount(self._red_mask) < self._red_count_target:
                break
            message = WheelsCmdStamped(vel_left=self._base_velocity, vel_right=self._base_velocity)
            dtros._wheels_publisher.publish(message)
            rate.sleep()

class Stop(FinalBehaviorMainTask):
    def __init__(self, stop_time=1):
        self._stop_time = stop_time
        self._stop_start_stamp = None

    def onStart(self, dtros):
        pass

    def isTaskFinished(self):
        if time.time() - self._stop_start_stamp > self._stop_time:
                return True
        return False

    def runTask(self, dtros):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self._stop_start_stamp is None:
                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._wheels_publisher.publish(message)
                self._stop_start_stamp = time.time()
                rate.sleep()
                continue
            if self.isTaskFinished():
                break
            message = WheelsCmdStamped(vel_left=0, vel_right=0)
            dtros._wheels_publisher.publish(message)
            rate.sleep()

class TailUntilIntersection(LaneFollowUntilIntersection):
    def __init__(self, base_velocity, tailing_task, debug=True):
        super().__init__(base_velocity=base_velocity, debug=debug)
        self._tailing_task = tailing_task

    def runTask(self, dtros):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            if self._tailing_task.isTargetTooClose():
                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._wheels_publisher.publish(message)
                rate.sleep()
                continue

            if self.isTimeToStop():
                break

            integration_stored_updated, error_last_updated, message = PIDOperations.getForwardPIDWheelMsg(base_velocity=self._base_velocity, error_last=self._error_last, integration_stored=self._integration_stored, error=self._error, proportional_gain=0.0000002, derivative_gain=0.0000002, integral_gain=0.0000002, integral_saturation=500000)
            self._error_last = error_last_updated
            self._integration_stored = integration_stored_updated
            dtros._wheels_publisher.publish(message)
            rate.sleep()

class TailWithTimeout(TailUntilIntersection):
    def __init__(self, tailing_task, base_velocity=0.25, timeout=4):
        super().__init__(base_velocity=0.25, tailing_task=tailing_task)
        self._timeout = timeout

        self._start_time_stamp = None

    def runTask(self, dtros):
        self._start_time_stamp = time.time()
        super().runTask(dtros)
    
    def isTimeToStop(self):
        if self._start_time_stamp is not None and (time.time() - self._start_time_stamp) > self._timeout:
            return True
        return False

class TailingLeftTurn(TurnLeftTask):
    def __init__(self, tailing_task, angular_velocity, debug=True):
        super().__init__(angular_velocity=angular_velocity, radians=math.pi/3)
        self._tailing_task = tailing_task
        self._debug=debug

    def runTask(self, dtros):
        precision = self._precision
        tolerance = self._tolerance
        target_radian = self._radians
        angular_velocity = self._angular_velocity
        R = self._R

        if self._debug == True:
            msg = f""" Running a curve task ... target_angle : {target_radian}, R: {R}, precision: {precision}, tolerance: {tolerance} """
            rospy.loginfo(msg)

        rate = rospy.Rate(precision)

        v_l = (R - self._l*3) * angular_velocity
        v_r = (R + self._l*3) * angular_velocity

        while not rospy.is_shutdown():
            message = WheelsCmdStamped(vel_left=v_l, vel_right=v_r)
            total_change_angle = ( self._distance_right - self._distance_left ) / (2*self._l)

            if self._debug == True:
                msg = f""" total_change_angle: {total_change_angle}, target_radian {target_radian}"""
                rospy.loginfo(msg)
            
            if self._tailing_task.isTargetTooClose():
                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._wheels_publisher.publish(message)
                continue

            if abs(total_change_angle) >= (abs(target_radian) - tolerance):
                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._wheels_publisher.publish(message)
                break

            dtros._wheels_publisher.publish(message)
            rate.sleep()

        if self._debug == True:
            msg = f""" total_change_angle: {total_change_angle}, target_radian {target_radian}"""
            rospy.loginfo(msg)


class TailingRightTurn(TurnRightTask):
    def __init__(self, tailing_task, debug=False):
        super().__init__(radians=math.pi/2, angular_velocity=2)
        self._tailing_task = tailing_task
        self._debug=debug

    def runTask(self, dtros):
        precision = self._precision
        tolerance = self._tolerance
        target_radian = self._radians
        angular_velocity = self._angular_velocity
        R = self._R

        if self._debug == True:
            msg = f""" Running a curve task ... target_angle : {target_radian}, R: {R}, precision: {precision}, tolerance: {tolerance} """
            rospy.loginfo(msg)

        rate = rospy.Rate(precision)

        v_r = (R - self._l*3) * angular_velocity
        v_l = (R + self._l*3) * angular_velocity

        while not rospy.is_shutdown():
            message = WheelsCmdStamped(vel_left=v_l, vel_right=v_r)
            total_change_angle = ( self._distance_right - self._distance_left ) / (2*self._l)

            if self._debug == True:
                msg = f""" total_change_angle: {total_change_angle}, target_radian {target_radian}"""
                rospy.loginfo(msg)
            
            if self._tailing_task.isTargetTooClose():
                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._wheels_publisher.publish(message)
                continue

            if abs(total_change_angle) >= (abs(target_radian) - tolerance):
                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._wheels_publisher.publish(message)
                break

            dtros._wheels_publisher.publish(message)
            rate.sleep()

        if self._debug == True:
            msg = f""" total_change_angle: {total_change_angle}, target_radian {target_radian}"""
            rospy.loginfo(msg)

class FreezeUntilTargetIsFarEnough(Stop):
    def __init__(self, tailing_task, stop_time=3):
        super().__init__(stop_time=stop_time)
        self._tailing_task = tailing_task
    
    def isTaskFinished(self):
        if self._tailing_task._duckie_blue_mask is None:
            return False
        if MaskOperations.getActiveCount(self._tailing_task._duckie_blue_mask) > 2000:
            return False
        if time.time() - self._stop_start_stamp > self._stop_time:
            return True
        return False

class Tailing(FinalBehaviorMainTask):
    def __init__(self, detection_threshold=3000, debug=False):
        self._bridge = CvBridge()
        self._duckie_blue_mask = None
        self._duckiebot_last_seen = None
        self._detection_threshold = detection_threshold
        self._debug = debug

    def onStart(self, dtros):
        vehicle_name = os.environ["VEHICLE_NAME"]
        self._led_publisher = rospy.Publisher(f'/{vehicle_name}/led_emitter_node/led_pattern', LEDPattern, queue_size=10)
        
        vehicle_name = os.environ["VEHICLE_NAME"]
        self._raw_image_topic = f"/{vehicle_name}/camera_node/image/compressed"
        self._sub_grawr_image = rospy.Subscriber(self._raw_image_topic, CompressedImage, self.callback_raw_image)
        
    def callback_raw_image(self, msg):
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        # undistorted = ImageOperations.undistort(image)
        self._duckie_blue_mask = ImageOperations.getDuckiebotBlueMask(image)

        if MaskOperations.getActiveCount(self._duckie_blue_mask) > 1000:
            if MaskOperations.getActiveCenter(self._duckie_blue_mask)[0] > ImageOperations.getImageWidth(self._duckie_blue_mask) - (4 * ImageOperations.getImageWidth(self._duckie_blue_mask)) // 9:
                self._duckiebot_last_seen = "Right"
            elif MaskOperations.getActiveCenter(self._duckie_blue_mask)[0] < (4 * ImageOperations.getImageWidth(self._duckie_blue_mask)) // 9:
                self._duckiebot_last_seen = "Left"
            else:
                self._duckiebot_last_seen = "Straight"

        # if MaskOperations.getActiveCount(self._duckie_blue_mask) > 1000:
        #     led_msg = ColorOperations.getLedMessage(colorPattern=ColorPattern(frontLeft=Colors.Green, frontRight=Colors.Green, backLeft=Colors.Green, backRight=Colors.Green))
        #     if MaskOperations.getActiveCenter(self._duckie_blue_mask)[0] > ImageOperations.getImageWidth(self._duckie_blue_mask) - (4 * ImageOperations.getImageWidth(self._duckie_blue_mask)) // 9:
        #         self._duckiebot_last_seen = "Right"
        #     elif MaskOperations.getActiveCenter(self._duckie_blue_mask)[0] < (4 * ImageOperations.getImageWidth(self._duckie_blue_mask)) // 9:
        #         self._duckiebot_last_seen = "Left"
        #     else:
        #         self._duckiebot_last_seen = "Straight"
        # else:
        #     led_msg = ColorOperations.getLedMessage(colorPattern=ColorPattern(frontLeft=Colors.Red, frontRight=Colors.Red, backLeft=Colors.Red, backRight=Colors.Red))
        # led_msg = ColorOperations.getLedMessage(colorPattern=ColorPattern(frontLeft=Colors.Off, frontRight=Colors.Off, backLeft=Colors.Off, backRight=Colors.Off))
        # self._led_publisher.publish(led_msg)

        # if self._debug == True:
        #     cv2.imshow("Duckie Blue Mask", self._duckie_blue_mask)
        #     cv2.waitKey(1)

    def isTargetTooClose(self):
        return MaskOperations.getActiveCount(self._duckie_blue_mask) > self._detection_threshold
    
    def runTask(self, dtros):
        counter = 0
        while not rospy.is_shutdown():
            TailUntilIntersection(base_velocity=0.3, tailing_task=self).execute(dtros)
            counter += 1
            if counter >= 4:
                Stop(stop_time=3).execute(dtros)
                break
            FreezeUntilTargetIsFarEnough(tailing_task=self).execute(dtros)
            DriveOverRedline().execute(dtros)
            if self._duckiebot_last_seen == "Right":
                TailingRightTurn(tailing_task=self, debug=False).execute(dtros)
                continue
            elif self._duckiebot_last_seen == "Left":
                TailingLeftTurn(tailing_task=self, angular_velocity=2).execute(dtros)
            TailUntilDistance(target_distance=0.7, tailing_task=self).execute(dtros)

class LRTicksTask(FinalBehaviorMainTask):
    def __init__(self):
        super().__init__()
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._ticks_left = None
        self._ticks_right = None

        self._distance_left = 0
        self._distance_right = 0
        self._distance = 0
        self._resolution = 135
        self._radius = rospy.get_param(f'/{self._vehicle_name}/kinematics_node/radius', 0.0318)

    def onStart(self, dtros):
        left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"
        self._sub_left_ticks = rospy.Subscriber(left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self._sub_right_ticks = rospy.Subscriber(right_encoder_topic, WheelEncoderStamped, self.callback_right)
    
    def callback_left(self, data):
        if self._ticks_left is None:
            self._ticks_left = data.data
            return
        
        self._distance_left += 2*math.pi*self._radius*((data.data - self._ticks_left)/self._resolution)
        self._ticks_left = data.data
        self.update_distance()
    
    def callback_right(self, data):
        if self._ticks_right is None:
            self._ticks_right = data.data
            return
        
        self._distance_right += 2*math.pi*self._radius*((data.data - self._ticks_right)/self._resolution)
        self._ticks_right = data.data
        self.update_distance()

    def update_distance(self):
        self._distance = (self._distance_left + self._distance_right) / 2
    
    def runTask(self, dtros):
        return super().runTask(dtros)

class BlindStraight(LRTicksTask):
    def __init__(self, target_distance=0.5, tolerance=0.1, velocity=0.5):
        super().__init__()
        self._target_distance = target_distance
        self._tolerance = tolerance
        self._velocity = velocity
    
    def runTask(self, dtros):
        # publish received tick messages every 0.05 second (20 Hz)
        rate = rospy.Rate(10)
        goal = self._distance + self._target_distance
        
        while not rospy.is_shutdown():

            if (self._distance > goal - self._tolerance):
                break
            else:
                message = WheelsCmdStamped(vel_left=self._velocity, vel_right=self._velocity)
                dtros._wheels_publisher.publish(message)

            rate.sleep()

class AprilTagTask(RawImageTask):
    def __init__(self, debug=True, detection_sensitivity=300):
        super().__init__()
        self._debug = debug
        self._detector = dt_apriltags.Detector(families="tag36h11")
        self._detection_sensitivity = detection_sensitivity
        self._results = None
        self._undistort_gray = None

    def callback_raw_image(self, msg):
        super().callback_raw_image(msg)
        undistort_grayscale = ImageOperations.getGrayscale(ImageOperations.undistort(self._raw_image))

        self._undistort_gray = undistort_grayscale

        self._results = ImageOperations.getAprilDetectionResults(undistort_grayscale, self._detector)

    def runTask(self, dtros):
        return super().runTask(dtros)
    
class TagAlignmentTask(AprilTagTask):
    def __init__(self, stall, proportional_gain, derivative_gain, timeout=5, integral_gain=0, integral_saturation=0, debug=True, detection_sensitivity=300):
        super().__init__(detection_sensitivity=detection_sensitivity, debug=debug)
        self._stall = stall
        self._error = 0
        self._error_last = 0
        self._integration_stored = 0
        self._integral_gain = integral_gain
        self._integral_saturation = integral_saturation
        self._proportional_gain = proportional_gain
        self._derivative_gain = derivative_gain
        self._timeout=timeout
        self._start_time_stamp = None
        
    def callback_raw_image(self, msg):
        super().callback_raw_image(msg)
        
        for r in self._results:
            if r.tag_id == self._stall.value:
                self._error = r.center[0] - ImageOperations.getCenterAxisX(self._undistort_gray)

                self._undistort_gray = ImageOperations.getAnnotateImage(self._undistort_gray, r)
            
        # cv2.imshow("Detection", self._undistort_gray)
        # cv2.waitKey(1)        

    def runTask(self, dtros):
        rate = rospy.Rate(10)  # 10 Hz control loop
        pulse_duration = 0.2   # seconds
        pause_duration = 0.3   # seconds
        last_pulse_time = 0
        pulsing = True

        while not rospy.is_shutdown():
            current_time = time.time()

            if self._start_time_stamp is None:
                self._start_time_stamp = current_time

            if (current_time - self._start_time_stamp) > self._timeout:
                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._wheels_publisher.publish(message)
                break

            # Compute control
            integration_stored_updated, error_last_updated, message = PIDOperations.getRotatePIDWheelMsg(
                error_last=self._error_last,
                integration_stored=self._integration_stored,
                error=self._error,
                proportional_gain=self._proportional_gain,
                derivative_gain=self._derivative_gain,
                integral_gain=self._integral_gain,
                integral_saturation=self._integral_saturation
            )
            self._error_last = error_last_updated
            self._integration_stored = integration_stored_updated

            if pulsing:
                if self._error > 0:
                    message = WheelsCmdStamped(vel_left=0.1, vel_right=-0.1)
                elif self._error < 0:
                    message = WheelsCmdStamped(vel_left=-0.1, vel_right=0.1)
                else:
                    message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._wheels_publisher.publish(message)
            else:
                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._wheels_publisher.publish(message)

            # Toggle pulse state
            if pulsing and (current_time - last_pulse_time) > pulse_duration:
                pulsing = False
                last_pulse_time = current_time
            elif not pulsing and (current_time - last_pulse_time) > pause_duration:
                pulsing = True
                last_pulse_time = current_time

            rate.sleep()

class LaneFollowUntilX(LaneFollowing):
    def __init__(self, base_velocity, debug=True):
        super().__init__(base_velocity=base_velocity, debug=debug)

    def isTimeToStop(self):
        raise Exception("isTimeToStop() must be overriden")

class LaneFollowUntilTimeout(LaneFollowUntilX):
    def __init__(self, base_velocity, timeout, debug=True):
        super().__init__(base_velocity=base_velocity, debug=debug)
        self._timeout = timeout
        self._start_time_stamp = None
    
    def onStart(self, dtros):
        super().onStart(dtros)
        self._start_time_stamp = time.time()
    
    def isTimeToStop(self):
        if (time.time() - self._start_time_stamp) > self._timeout:
            return True
        return False


class TailUntilX(LaneFollowUntilX):
    def __init__(self, base_velocity, tailing_task, debug=True):
        super().__init__(base_velocity=base_velocity, debug=debug)
        self._tailing_task = tailing_task

    def runTask(self, dtros):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            if self._tailing_task.isTargetTooClose():
                message = WheelsCmdStamped(vel_left=0, vel_right=0)
                dtros._wheels_publisher.publish(message)
                rate.sleep()
                continue

            if self.isTimeToStop():
                break

            integration_stored_updated, error_last_updated, message = PIDOperations.getForwardPIDWheelMsg(base_velocity=self._base_velocity, error_last=self._error_last, integration_stored=self._integration_stored, error=self._error, proportional_gain=0.0000002, derivative_gain=0.0000002, integral_gain=0.0000002, integral_saturation=500000)
            self._error_last = error_last_updated
            self._integration_stored = integration_stored_updated
            dtros._wheels_publisher.publish(message)
            rate.sleep()

    def isTimeToStop(self):
        raise Exception("isTimeToStop() must be overriden")

class TailUntilDistance(TailUntilX):
    def __init__(self, tailing_task, target_distance, base_velocity=0.25):
        super().__init__(base_velocity=base_velocity, tailing_task=tailing_task)
        self._target_distance = target_distance
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._ticks_left = None
        self._ticks_right = None

        self._distance_left = 0
        self._distance_right = 0
        self._distance = 0
        self._resolution = 135
        self._radius = rospy.get_param(f'/{self._vehicle_name}/kinematics_node/radius', 0.0318)

    def onStart(self, dtros):
        super().onStart(dtros)
        left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"
        self._sub_left_ticks = rospy.Subscriber(left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self._sub_right_ticks = rospy.Subscriber(right_encoder_topic, WheelEncoderStamped, self.callback_right)
    
    def callback_left(self, data):
        if self._ticks_left is None:
            self._ticks_left = data.data
            return
        
        self._distance_left += 2*math.pi*self._radius*((data.data - self._ticks_left)/self._resolution)
        self._ticks_left = data.data
        self.update_distance()
    
    def callback_right(self, data):
        if self._ticks_right is None:
            self._ticks_right = data.data
            return
        
        self._distance_right += 2*math.pi*self._radius*((data.data - self._ticks_right)/self._resolution)
        self._ticks_right = data.data
        self.update_distance()

    def update_distance(self):
        self._distance = (self._distance_left + self._distance_right) / 2

    def isTimeToStop(self):
        if self._distance > self._target_distance:
            return True
        return False

class LaneFollowUntilDistance(LaneFollowUntilX):
    def __init__(self, target_distance, base_velocity=0.25):
        super().__init__(base_velocity=base_velocity)
        self._target_distance = target_distance
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._ticks_left = None
        self._ticks_right = None

        self._distance_left = 0
        self._distance_right = 0
        self._distance = 0
        self._resolution = 135
        self._radius = rospy.get_param(f'/{self._vehicle_name}/kinematics_node/radius', 0.0318)

    def onStart(self, dtros):
        super().onStart(dtros)
        left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"
        self._sub_left_ticks = rospy.Subscriber(left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self._sub_right_ticks = rospy.Subscriber(right_encoder_topic, WheelEncoderStamped, self.callback_right)
    
    def callback_left(self, data):
        if self._ticks_left is None:
            self._ticks_left = data.data
            return
        
        self._distance_left += 2*math.pi*self._radius*((data.data - self._ticks_left)/self._resolution)
        self._ticks_left = data.data
        self.update_distance()
    
    def callback_right(self, data):
        if self._ticks_right is None:
            self._ticks_right = data.data
            return
        
        self._distance_right += 2*math.pi*self._radius*((data.data - self._ticks_right)/self._resolution)
        self._ticks_right = data.data
        self.update_distance()

    def update_distance(self):
        self._distance = (self._distance_left + self._distance_right) / 2

    def isTimeToStop(self):
        if self._distance > self._target_distance:
            return True
        return False

class LeftRightTagTask(RawImageTask):
    def __init__(self, debug=True, detection_sensitivity=300):
        super().__init__()
        self._debug = debug
        self._detector = dt_apriltags.Detector(families="tag36h11")
        self._detection_sensitivity = detection_sensitivity
        self._recent_tag_detected = None

    def callback_raw_image(self, msg):
        super().callback_raw_image(msg)
        # if self._debug == True:
        #     cv2.imshow("April Tag detection", undistort_grayscale)
        #     cv2.waitKey(1)
    
    def runTask(self, dtros):
        counter = 0
        while not rospy.is_shutdown():
            if counter >= 2:
                break
            LaneFollowUntilIntersection(base_velocity=0.3).execute(dtros)
            Stop(stop_time=3).execute(dtros)

            BackwardsWithDelay(timeout=3).execute(dtros)

            undistort_grayscale = ImageOperations.getGrayscale(ImageOperations.undistort(self._raw_image))
            results = ImageOperations.getAprilDetectionResults(undistort_grayscale, self._detector)
            for r in results:
                if (r.tag_id == Tag.LEFT.value or r.tag_id == Tag.RIGHT.value) and TagROperations.getPerimeterInPixels(r) > 100:
                    self._recent_tag_detected = Tag.LEFT if r.tag_id == Tag.LEFT.value else Tag.RIGHT 
            LaneFollowUntilIntersection(base_velocity=0.3).execute(dtros)
            Stop(stop_time=3).execute(dtros)

            DriveOverRedline(red_count_target=1000).execute(dtros)
            Stop(stop_time=0).execute(dtros)

            if self._recent_tag_detected == Tag.LEFT:
                left_tag_seen = True
                TurnLeftTask(angular_velocity=2, radians=math.pi/3).execute(dtros)
                LaneFollowUntilDistance(target_distance=0.7).execute(dtros)
            elif self._recent_tag_detected == Tag.RIGHT:
                right_tag_seen = True
                TurnRightTask(radians=math.pi/2, angular_velocity=2, R=0.25).execute(dtros)
                Stop(stop_time=0).execute(dtros)
            counter += 1

class BackwardsWithDelay(FinalBehaviorMainTask):
    def __init__(self, timeout=2):
        self._timeout = timeout
        self._start_time_stamp = None
    
    def onStart(self, dtros):
        self._start_time_stamp = time.time()
    
    def runTask(self, dtros):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if (time.time() - self._start_time_stamp) > self._timeout:
                break
            message = WheelsCmdStamped(vel_left=-0.3, vel_right=-0.3)
            dtros._wheels_publisher.publish(message)
            rate.sleep()

class TailingWithBluePID(FinalBehaviorMainTask):
    def __init__(self, base_velocity=0.3, scale=1, debug=False):
        # PID states for lane-following
        self._lane_error_last = 0
        self._lane_integration = 0
        # PID states for blue-following
        self._blue_error_last = 0
        self._blue_integration = 0
        
        # Detection flag
        self._blue_detected = False

        # Current error
        self._error = 0

        # PID gains & limits for lane-following
        self._kp_lane = 2e-7
        self._kd_lane = 2e-7
        self._ki_lane = 2e-7
        self._i_sat_lane = 500_000
        # PID gains & limits for blue-following
        self._kp_blue = 1e-4
        self._kd_blue = 1e-4
        self._ki_blue = 5e-6
        self._i_sat_blue = 100_000

        self._base_velocity = base_velocity
        self._scale = scale  # downscale factor for performance
        self._debug = debug
        self._bridge = CvBridge()

    def onStart(self, dtros):
        vehicle = os.environ['VEHICLE_NAME']
        topic = f"/{vehicle}/camera_node/image/compressed"
        self._sub_raw_image = rospy.Subscriber(topic, CompressedImage, self.callback_raw_image)

    def callback_raw_image(self, msg):
        # Convert ROS image to OpenCV
        raw = self._bridge.compressed_imgmsg_to_cv2(msg)
        blur = cv2.GaussianBlur(raw,(11,11),0)
        # Undistort full resolution
        undist_full = ImageOperations.undistort(blur)
        # Bird's-eye warp full resolution
        warped_full = ImageOperations.getHomography(undist_full)
        # Downscale images
        # undist = cv2.resize(undist_full, None, fx=self._scale, fy=self._scale)
        # warped = cv2.resize(warped_full, None, fx=self._scale, fy=self._scale)

        # Compute lane error on warped image
        target_x_white = int(489 * self._scale)
        target_x_yellow = int(100 * self._scale)
        mask_w = ImageOperations.getWhiteMask(warped_full)
        mask_y = ImageOperations.getYellowMask(warped_full)
        err_w = MaskOperations.computeErrorInAxisX(mask_w, target_x=target_x_white, pixel_value=1)
        err_y = MaskOperations.computeErrorInAxisX(mask_y, target_x=target_x_yellow, pixel_value=1)
        lane_error = err_w + err_y

        # Compute blue robot error on downscaled undistorted image
        mask_b = ImageOperations.getDuckiebotBlueMask(undist_full)
        cx_b = MaskOperations.getActiveCenterX(mask_b)
        center_x = ImageOperations.getCenterAxisX(undist_full)
        detected = (cx_b != -math.inf)
        blue_error = (cx_b - center_x) * detected
        self._blue_detected = bool(detected)
        self._error = blue_error + lane_error * (1.0 - detected)
        # self._error = lane_error

        if self._debug:
            state = 'TAIL_BLUE' if self._blue_detected else 'FOLLOW_LANE'
            rospy.loginfo(f"[{self.__class__.__name__}] state={state}, error={self._error:.1f}, cx_b={cx_b:.1f}")
            cv2.imshow("Undistort Blue Mask", mask_b)
            cv2.imshow("raw", raw)
            cv2.waitKey(1)

    def isTimeToStop(self):
        return False

    def runTask(self, dtros):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.isTimeToStop():
                break

            # Adjust base speed if tailing
            speed = self._base_velocity * (0.8 if self._blue_detected else 1.0)

            # Run the appropriate PID
            if self._blue_detected:
                i_new, e_new, cmd = PIDOperations.getForwardPIDWheelMsg(
                    base_velocity=speed,
                    error_last=self._blue_error_last,
                    integration_stored=self._blue_integration,
                    error=self._error,
                    proportional_gain=self._kp_blue,
                    derivative_gain=self._kd_blue,
                    integral_gain=self._ki_blue,
                    integral_saturation=self._i_sat_blue,
                )
                self._blue_integration = i_new
                self._blue_error_last = e_new
            else:
                i_new, e_new, cmd = PIDOperations.getForwardPIDWheelMsg(
                    base_velocity=speed,
                    error_last=self._lane_error_last,
                    integration_stored=self._lane_integration,
                    error=self._error,
                    proportional_gain=self._kp_lane,
                    derivative_gain=self._kd_lane,
                    integral_gain=self._ki_lane,
                    integral_saturation=self._i_sat_lane,
                )
                self._lane_integration = i_new
                self._lane_error_last = e_new

            # dtros._wheels_publisher.publish(cmd)
            rate.sleep()
