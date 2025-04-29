#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped, LEDPattern
import math
import time
from std_msgs.msg import ColorRGBA, Header


class DPATH(DTROS):
    def __init__(self, node_name, tasks):
        super(DPATH, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._radius = rospy.get_param(f'/{self._vehicle_name}/kinematics_node/radius', 0.0318)

        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        state_topic = f"/{self._vehicle_name}/state"
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
        self._led_publisher = rospy.Publisher(f'/{self._vehicle_name}/led_emitter_node/led_pattern', LEDPattern, queue_size=10)
        self._state_publisher = rospy.Publisher(state_topic, String, queue_size=1)

        
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"
        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)

        self._ticks_left = None
        self._ticks_right = None

        self._distance_left = 0
        self._distance_right = 0
        
        self._l = 0.050

        self._resolution = 135

        self._tasks = tasks

        self._start = rospy.Time.now()

    def callback_left(self, data):
        if self._ticks_left is None:
            self._ticks_left = data.data
            return
        
        self._distance_left += 2*math.pi*self._radius*((data.data - self._ticks_left)/self._resolution)
        self._ticks_left = data.data
    
    def callback_right(self, data):
        if self._ticks_right is None:
            self._ticks_right = data.data
            return
        
        self._distance_right += 2*math.pi*self._radius*((data.data - self._ticks_right)/self._resolution)
        self._ticks_right = data.data

    def run(self):
        for task in self._tasks:

            if not isinstance(task,Task):
                raise ValueError("task not recognized")
            
            self._distance_right = 0
            self._distance_left = 0

            task.execute(self)
            
        rospy.signal_shutdown(reason="tasks complete")
    
    def on_shutdown(self):
        self._end = rospy.Time.now()
        duration = (self._end - self._start).to_sec()
        rospy.loginfo(f"Execution time: {duration}")
        Shutdown().execute(self)


class Task():
    def __init__(self, throttle=0, precision=0, tolerance=0, led_color=[0.0,1.0,0.0]):
        self._throttle = throttle
        self._precision = precision
        self._tolerance = tolerance
        self._led_color = led_color

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
        dtros._state_publisher.publish(state)
        self.runTask(dtros)

class Stop(Task):
    def __init__(self, throttle=0, precision=0, tolerance=0, stop_time=5, led_color=[1.0,0.0,0.0]):
        super().__init__(throttle, precision, tolerance, led_color)
        self._stop_time = stop_time
    
    def execute(self, dtros):
        state = "Stopping"
        dtros._state_publisher.publish(state)
        self.runTask(dtros)

    def runTask(self, dtros):
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
        dtros._state_publisher.publish(state)

class RotateTask(MoveTask):
    def __init__(self, throttle=0.5, precision=50, tolerance=0.5, radian=-math.pi/2):
        super().__init__(throttle, precision, tolerance)
        if radian > 2*math.pi or radian < -2*math.pi:
            raise ValueError("radian must be between 2pi and -2pi")
        
        self._angle_rotation_radians = radian
        self._direction_left = 1 if radian <= 0 else -1
        self._direction_right = -1 if radian <= 0 else 1
    
    def get_angle_rotation_radians(self):
        return self._angle_rotation_radians
    
    def get_direction_of_right_wheel(self):
        return self._direction_right
    
    def get_direction_of_left_wheel(self):
        return self._direction_left
    
    def runTask(self, dtros):
        dir_r_wheel = self.get_direction_of_right_wheel()
        dir_l_wheel = self.get_direction_of_left_wheel()
        target_radian = self.get_angle_rotation_radians()
        throttle = self.get_throttle()
        precision = self.get_precision()
        tolerance = self.get_tolerance()

        msg = f""" Running a rotate task ... target radian : {target_radian}"""
        rospy.loginfo(msg)

        msg = f""" Led Color: {self.get_led_color()}"""
        rospy.loginfo(msg)

        rate = rospy.Rate(precision)
        message = WheelsCmdStamped(vel_left=throttle*dir_l_wheel, vel_right=throttle*dir_r_wheel)

        while not rospy.is_shutdown():
            total_change_angle = ( dtros._distance_right - dtros._distance_left ) / (2*dtros._l)
            msg = f""" total_change_angle: {total_change_angle}, target_radian {target_radian}"""
            rospy.loginfo(msg)
            if abs(total_change_angle) >= (abs(target_radian) - tolerance):
                break
            self.add_header(message)
            dtros._publisher.publish(message)
            rate.sleep()

        msg = f""" total_change_angle: {total_change_angle}, target_radian {target_radian}"""
        rospy.loginfo(msg)
    
class CurveTask(MoveTask):
    def __init__(self, throttle=0.8, precision=40, tolerance=0.04, R=0.29, target_radian=-math.pi/2, right_offset=0.25):
        super().__init__(throttle, precision, tolerance)
        if target_radian > 2*math.pi or target_radian < -2*math.pi:
            raise ValueError("radian must be between 2pi and -2pi")
        self._R = R
        self._target_radian = target_radian
        self._right_offset = right_offset
    
    def get_target_radian(self):
        return self._target_radian
    
    def get_R(self):
        return self._R

    def get_right_offset(self):
        return self._right_offset
    
    def runTask(self, dtros):
        target_radian = self.get_target_radian()
        R = self.get_R()
        right_offset = self.get_right_offset()
        throttle = self.get_throttle()
        precision = self.get_precision()
        tolerance = self.get_tolerance()
    
        msg = f""" Running a curve task ... target_angle : {target_radian}, R: {R}, right_offset: {right_offset}, throttle:{throttle}, precision: {precision}, tolerance: {tolerance} """
        rospy.loginfo(msg)

        msg = f""" Led Color: {self.get_led_color()}"""
        rospy.loginfo(msg)

        rate = rospy.Rate(precision)
        furthest_wheel = R + dtros._l
        closest_wheel = R - dtros._l
        distance_ratio = closest_wheel / furthest_wheel
        msg = f""" Distance Ratio: {distance_ratio} """
        rospy.loginfo(msg)
        message = WheelsCmdStamped(vel_left=throttle*1, vel_right=throttle*(distance_ratio)-right_offset)

        while not rospy.is_shutdown():
            total_change_angle = ( dtros._distance_right - dtros._distance_left ) / (2*dtros._l)
            msg = f""" total_change_angle: {total_change_angle}, target_radian {target_radian}"""
            rospy.loginfo(msg)
            if abs(total_change_angle) >= (abs(target_radian) - tolerance):
                break

            self.add_header(message)
            dtros._publisher.publish(message)
            rate.sleep()

        msg = f""" total_change_angle: {total_change_angle}, target_radian {target_radian}"""
        rospy.loginfo(msg)



class StraightTask(MoveTask):
    def __init__(self, throttle=0.8, precision=40, tolerance=0.1, distance=1.2, right_offset=0, left_offset=0):
        super().__init__(throttle, precision, tolerance)
        self._target_distance = distance
        self._right_offset = right_offset
        self._left_offset = left_offset

    def get_target_distance(self):
        return self._target_distance

    def get_right_offset(self):
        return self._right_offset

    def get_left_offset(self):
        return self._left_offset
    
    def runTask(self, dtros):
        target_distance = self.get_target_distance()
        throttle = self.get_throttle()
        precision = self.get_precision()
        tolerance = self.get_tolerance()
        right_offset = self.get_right_offset()
        left_offset = self.get_left_offset()

        msg = f""" Running a straight task ... target distance {target_distance} """
        rospy.loginfo(msg)

        msg = f""" Led Color: {self.get_led_color()}"""
        rospy.loginfo(msg)

        rate = rospy.Rate(precision)
        message = WheelsCmdStamped(vel_left=throttle*1+left_offset, vel_right=throttle*1+right_offset)

        while not rospy.is_shutdown():
            distance_so_far = ( dtros._distance_left + dtros._distance_right ) / 2
            msg = f""" distance_so_far: {distance_so_far}, target_distance :{target_distance}"""
            rospy.loginfo(msg)
            if distance_so_far >= ( target_distance - tolerance ):
                break

            self.add_header(message)
            dtros._publisher.publish(message)
            rate.sleep()
            msg = f""" distance_so_far: {distance_so_far}, target_distance :{target_distance}"""
        rospy.loginfo(msg)

        
if __name__ == "__main__":

    # Calibration for this program
    # baseline: 0.1
    # calibration_time: 2025-02-11-18-05-50
    # gain: 3.0
    # k: 27.0
    # limit: 1.0
    # omega_max: 8.0
    # radius: 0.0318
    # trim: 0.2
    # v_max: 1.0

    try:
        straight_left_offset = 0.01
        straight_throttle = 0.7
        stop_time = 3
        tasks = [
            Stop(stop_time=5),
            StraightTask(throttle=0.5, precision=40, tolerance=0.1,distance=1.2, left_offset=0), 
            Stop(stop_time=stop_time),
            RotateTask(throttle=0.4, precision=40, tolerance=0.4, radian=-math.pi/2), 
            Stop(stop_time=stop_time),
            StraightTask(throttle=straight_throttle, precision=40, tolerance=0.1, distance=0.92, left_offset=0.03), 
            Stop(stop_time=stop_time),
            CurveTask(throttle=0.7, precision=40, tolerance=0.04, R=0.29, target_radian=-math.pi/2, right_offset=0.25), 
            Stop(stop_time=stop_time),
            StraightTask(throttle=straight_throttle, precision=40, tolerance=0.1, distance=0.61, left_offset=0.03), 
            Stop(stop_time=stop_time),
            CurveTask(throttle=0.8, precision=40, tolerance=0.04, R=0.29, target_radian=-math.pi/2, right_offset=0.25), 
            Stop(stop_time=stop_time),
            StraightTask(throttle=straight_throttle, precision=40, tolerance=0.1, distance=0.92, left_offset=0.03),
            Stop(stop_time=stop_time),
            RotateTask(throttle=0.4, precision=50, tolerance=0.4, radian=-math.pi/2),
            Stop(stop_time=5),
        ]
        node = DPATH(node_name="rotate_node", tasks=tasks)
        node.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass