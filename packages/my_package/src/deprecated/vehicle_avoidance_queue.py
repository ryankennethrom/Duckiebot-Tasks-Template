#!/usr/bin/env python3

import os
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped, LEDPattern
from std_msgs.msg import ColorRGBA, Header
import cv2
from cv_bridge import CvBridge
import time
import types

class DPATH(DTROS):
    def __init__(self, node_name, tasks):
        super(DPATH, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._radius = rospy.get_param(f'/{self._vehicle_name}/kinematics_node/radius', 0.0318)
        
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

        self._ticks_left = None
        self._ticks_right = None

        self._distance_left = 0
        self._distance_right = 0
        
        self._l = 0.050

        self._resolution = 135

        self._tasks = tasks

        self._start = rospy.Time.now()

    # def callback_left(self, data):
    #     if self._ticks_left is None:
    #         self._ticks_left = data.data
    #         return
        
    #     self._distance_left += 2*math.pi*self._radius*((data.data - self._ticks_left)/self._resolution)
    #     self._ticks_left = data.data
    
    # def callback_right(self, data):
    #     if self._ticks_right is None:
    #         self._ticks_right = data.data
    #         return
        
    #     self._distance_right += 2*math.pi*self._radius*((data.data - self._ticks_right)/self._resolution)
    #     self._ticks_right = data.data

    def run(self):
        for task in self._tasks:

            if not isinstance(task,Task):
                raise ValueError("task not recognized")
            
            task.setup(self)
            task.execute(self)
            task.cleanup(self)
            
        rospy.signal_shutdown(reason="tasks complete")

    def on_shutdown(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop)

class Task():
    def __init__(self, throttle=0, precision=0, tolerance=0, led_color=[0.0,1.0,0.0]):
        self._throttle = throttle
        self._precision = precision
        self._tolerance = tolerance
        self._led_color = led_color

    def setup(self, *args):
        pass

    def cleanup(self, *args):
        pass

    def get_led_color(self):
        return self._led_color
    
    def get_tolerance(self):
        return self._tolerance
    
    def get_precision(self):
        return self._precision
    
    def get_throttle(self):
        return self._throttle
    
    def execute(self, dtros):
        self.runTask(dtros)
    
    def runTask(self, dtros):
        pass

    def create_led_msg(self, colors):
        led_msg = LEDPattern()

        for i in range(5):
            rgba = ColorRGBA()
            rgba.r = colors[0]
            rgba.g = colors[1]
            rgba.b = colors[2]
            rgba.a = 1.0
            led_msg.rgb_vals.append(rgba)

        return led_msg
    
    def add_header(self, message):
        h = Header()
        h.stamp = rospy.Time.now()
        message.header = h

class MoveTask(Task):
    def __init__(self, throttle, precision, tolerance, led_color=[0.0,1.0,0.0]):
        super().__init__(throttle, precision, tolerance, led_color)
    
    def execute(self, dtros):
        state = "Moving"
        self.runTask(dtros)

class Stop(Task):
    def __init__(self, throttle=0, precision=0, tolerance=0, stop_time=5, led_color=[1.0,0.0,0.0]):
        super().__init__(throttle, precision, tolerance, led_color)
        self._stop_time = stop_time
    
    def execute(self, dtros):
        state = "Stopping"
        self.runTask(dtros)

    def runTask(self, dtros):
        print("RUNNING: Stop")
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self.add_header(stop)
        dtros._publisher.publish(stop)
        msg = f""" Led Color: {self._led_color}"""
        rospy.loginfo(msg)
        time.sleep(self._stop_time)

class Shutdown(Task):
    def __init__(self, throttle=0, precision=0, tolerance=0, led_color=[0.0,0.0,0.0]):
        super().__init__(throttle, precision, tolerance, led_color)
    
    def execute(self, dtros):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self.add_header(stop)
        dtros._publisher.publish(stop)
        state = "Exiting"

class VehicleAvoidance(MoveTask):
    def __init__(self, 
                 node_name="", 
                 proportional_gain=0.0000002, 
                 derivative_gain=0.0000002,
                 integral_gain=0.0000002,
                 velocity=0.3,
                 integral_saturation=500000,
                 yellow_target_x = 489,
                 white_target_x = 100):
        self.proportional_gain = proportional_gain
        self.derivative_gain = derivative_gain
        self.integral_gain = integral_gain
        self.vel = velocity
        self.integral_saturation = integral_saturation

        self._error = 20
        self._error_last = self._error
        self._integration_stored = 0

        self.yellow_target_x = yellow_target_x
        self.white_target_x = white_target_x

    def setup(self, dtros):
        dtros._camera_topic = f"/{dtros._vehicle_name}/camera_node/undistorted_image/compressed"
        dtros._blue_detection_topic = f"/{dtros._vehicle_name}/camera_node/blue_image_mask/compressed"
        dtros._bridge = CvBridge()

        # add functions as a method of dtros
        dtros.callback = types.MethodType(self.callback, dtros)
        dtros.getUpdate = types.MethodType(self.getUpdate, dtros)
        dtros.compute_error = types.MethodType(self.compute_error, dtros)
        dtros.blue_detection_callback = types.MethodType(self.blue_detection_callback, dtros)

        dtros.sub = rospy.Subscriber(dtros._camera_topic, CompressedImage, dtros.callback)
        dtros.blue_mask_sub = rospy.Subscriber(dtros._blue_detection_topic, CompressedImage, dtros.blue_detection_callback)

        dtros.proportional_gain = self.proportional_gain
        dtros.derivative_gain = self.derivative_gain
        dtros.integral_gain = self.integral_gain
        dtros.vel = self.vel
        dtros.integral_saturation = self.integral_saturation

        dtros._error = 20
        dtros._error_last = self._error
        dtros._integration_stored = 0

        dtros.yellow_target_x = self.yellow_target_x
        dtros.white_target_x = self.white_target_x

        dtros._crosswalk_detected = False
        dtros._duckie_detected = False
    
    def cleanup(self, dtros):
        # need to cleanup only pubs and subs
        del dtros.sub
        del dtros.blue_mask_sub

    def blue_detection_callback(self, dtros, *args):
        msg = args[0]
        # msg is already a mask
        mask_blue = dtros._bridge.compressed_imgmsg_to_cv2(msg)

        # Find the coordinates of active (non-zero) pixels
        y_coords, x_coords = np.where(mask_blue > 0)  # y, x positions of active pixels

        if len(x_coords) == 0:
            # print("Detecting nothing")
            return

        # Calculate the variance of the x coordinates
        x_variance = np.var(x_coords)

        # Calculate the magnitude (number of active blue pixels)
        magnitude = len(x_coords)

        # print("Magnitude : " + str(magnitude) + ", Variance : " + str(x_variance))

        if (magnitude >= 5000):
            if(x_variance > 10000):
                # print("Detecting crosswalk")
                dtros._crosswalk_detected = True
                dtros._duckie_detected = False
            else:
                print("Detecting duckiebot")
                dtros._duckie_detected = True
                dtros._crosswalk_detected = False
        else:
            # print("Detecting nothing")
            dtros._duckie_detected = False
            dtros._crosswalk_detected = False

    def callback(self, dtros, *args):
        msg = args[0]
        # convert JPEG bytes to CV image
        image = dtros._bridge.compressed_imgmsg_to_cv2(msg)
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

        lower_blue = np.array([100, 150, 0], dtype=np.uint8)  # Lower bound for blue
        upper_blue = np.array([140, 255, 255], dtype=np.uint8)  # Upper bound for blue
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        yellow_error = self.compute_error(mask=mask_yellow, target_x=dtros.yellow_target_x)
        white_error = self.compute_error(mask=mask_white, target_x=dtros.white_target_x)
        blue_error = self.compute_error(mask=mask_blue, target_x=589, pixel_value=35000)

        if yellow_error > 100000 and white_error < -100000:
            self._error = yellow_error + abs(white_error) 

        elif yellow_error < -100000 and white_error > 100000:
            self._error = yellow_error + -1*white_error
        else:
            self._error = yellow_error + white_error
        

        dtros._error += blue_error
        # print(blue_error)

    def getUpdate(self, dtros):
        P = dtros._error*dtros.proportional_gain
        errorRateOfChange = dtros._error - dtros._error_last
        D = dtros.derivative_gain * errorRateOfChange
        integration_stored_update = dtros._integration_stored + (dtros._error)
        dtros._integration_stored = (integration_stored_update) if abs(integration_stored_update) <= dtros.integral_saturation else (integration_stored_update/integration_stored_update)*dtros.integral_saturation
        I = dtros.integral_gain * dtros._integration_stored

        dtros._error_last = dtros._error

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

    def runTask(self, dtros):
        print("RUNNING: VehicleAvoidance")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            correctionUpdate = dtros.getUpdate()
        
            if correctionUpdate < 0:
                message = WheelsCmdStamped(vel_left=self.vel, vel_right=self.vel+abs(correctionUpdate))
            elif correctionUpdate > 0:
                message = WheelsCmdStamped(vel_left=self.vel+abs(correctionUpdate), vel_right=self.vel)
            else:
                message = WheelsCmdStamped(vel_left=self.vel, vel_right=self.vel)
            
            dtros._publisher.publish(message)
            
            rate.sleep()

class FindDuckieBot(MoveTask):
    def __init__(self, 
                 node_name="", 
                 proportional_gain=0.0000002, 
                 derivative_gain=0.0000002,
                 integral_gain=0.0000002,
                 velocity=0.3,
                 integral_saturation=500000,
                 yellow_target_x = 100,
                 white_target_x = 489):
        self.proportional_gain = proportional_gain
        self.derivative_gain = derivative_gain
        self.integral_gain = integral_gain
        self.vel = velocity
        self.integral_saturation = integral_saturation

        self._error = 20
        self._error_last = self._error
        self._integration_stored = 0

        self.yellow_target_x = yellow_target_x
        self.white_target_x = white_target_x

    def setup(self, dtros):
        dtros._camera_topic = f"/{dtros._vehicle_name}/camera_node/undistorted_image/compressed"
        dtros._blue_detection_topic = f"/{dtros._vehicle_name}/camera_node/blue_image_mask/compressed"
        dtros._bridge = CvBridge()

        # add functions as a method of dtros
        dtros.callback = types.MethodType(self.callback, dtros)
        dtros.getUpdate = types.MethodType(self.getUpdate, dtros)
        dtros.compute_error = types.MethodType(self.compute_error, dtros)
        dtros.blue_detection_callback = types.MethodType(self.blue_detection_callback, dtros)

        dtros.sub = rospy.Subscriber(dtros._camera_topic, CompressedImage, dtros.callback)
        dtros.blue_mask_sub = rospy.Subscriber(dtros._blue_detection_topic, CompressedImage, dtros.blue_detection_callback)

        dtros.proportional_gain = self.proportional_gain
        dtros.derivative_gain = self.derivative_gain
        dtros.integral_gain = self.integral_gain
        dtros.vel = self.vel
        dtros.integral_saturation = self.integral_saturation

        dtros._error = 20
        dtros._error_last = self._error
        dtros._integration_stored = 0

        dtros.yellow_target_x = self.yellow_target_x
        dtros.white_target_x = self.white_target_x

        dtros._crosswalk_detected = False
        dtros._duckie_detected = False

    def cleanup(self, dtros):
        # need to cleanup only pubs and subs
        del dtros.sub
        del dtros.blue_mask_sub

    def blue_detection_callback(self, dtros, *args):
        msg = args[0]
        # msg is already a mask
        mask_blue = dtros._bridge.compressed_imgmsg_to_cv2(msg)

        # Find the coordinates of active (non-zero) pixels
        y_coords, x_coords = np.where(mask_blue > 0)  # y, x positions of active pixels

        if len(x_coords) == 0:
            # print("Detecting nothing")
            return

        # Calculate the variance of the x coordinates
        x_variance = np.var(x_coords)

        # Calculate the magnitude (number of active blue pixels)
        magnitude = len(x_coords)

        # print("Magnitude : " + str(magnitude) + ", Variance : " + str(x_variance))

        if (magnitude >= 5000):
            if(x_variance > 10000):
                # print("Detecting crosswalk")
                dtros._crosswalk_detected = True
                dtros._duckie_detected = False
            else:
                print("Detecting duckiebot")
                dtros._duckie_detected = True
                dtros._crosswalk_detected = False
        else:
            # print("Detecting nothing")
            dtros._duckie_detected = False
            dtros._crosswalk_detected = False

    def callback(self, dtros, *args):
        msg = args[0]
        # convert JPEG bytes to CV image
        image = dtros._bridge.compressed_imgmsg_to_cv2(msg)
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

        lower_blue = np.array([100, 150, 0], dtype=np.uint8)  # Lower bound for blue
        upper_blue = np.array([140, 255, 255], dtype=np.uint8)  # Upper bound for blue
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        yellow_error = self.compute_error(mask=mask_yellow, target_x=dtros.yellow_target_x)
        white_error = self.compute_error(mask=mask_white, target_x=dtros.white_target_x)
        # blue_error = self.compute_error(mask=mask_blue, target_x=589, pixel_value=35000)

        if yellow_error > 100000 and white_error < -100000:
            self._error = yellow_error + abs(white_error) 

        elif yellow_error < -100000 and white_error > 100000:
            self._error = yellow_error + -1*white_error
        else:
            self._error = yellow_error + white_error
        

        # dtros._error += blue_error
        # print(blue_error)

    def getUpdate(self, dtros):
        P = dtros._error*dtros.proportional_gain
        errorRateOfChange = dtros._error - dtros._error_last
        D = dtros.derivative_gain * errorRateOfChange
        integration_stored_update = dtros._integration_stored + (dtros._error)
        dtros._integration_stored = (integration_stored_update) if abs(integration_stored_update) <= dtros.integral_saturation else (integration_stored_update/integration_stored_update)*dtros.integral_saturation
        I = dtros.integral_gain * dtros._integration_stored

        dtros._error_last = dtros._error

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

    def runTask(self, dtros):
        print("RUNNING: FindDuckieBot")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if dtros._duckie_detected:
                print("ENDINGGGGGGG")
                break
            correctionUpdate = dtros.getUpdate()
        
            if correctionUpdate < 0:
                message = WheelsCmdStamped(vel_left=self.vel, vel_right=self.vel+abs(correctionUpdate))
            elif correctionUpdate > 0:
                message = WheelsCmdStamped(vel_left=self.vel+abs(correctionUpdate), vel_right=self.vel)
            else:
                message = WheelsCmdStamped(vel_left=self.vel, vel_right=self.vel)
            
            dtros._publisher.publish(message)
            
            rate.sleep()


if __name__ == "__main__":
    try:
        tasks = [FindDuckieBot(), Stop(stop_time=3), FindDuckieBot(yellow_target_x=489, white_target_x=100)]
        node = DPATH(node_name="main", tasks=tasks)
        node.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass