#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
import math
import time

class Rotate(DTROS):
    def __init__(self, node_name, tasks, throttle, tolerance, precision):
        super(Rotate, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._radius = rospy.get_param(f'/{self._vehicle_name}/kinematics_node/radius', 0.0318)

        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
        
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"
        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)

        self._ticks_left = None
        self._ticks_right = None

        self._distance_left = 0
        self._distance_right = 0

        self._throttle = throttle
        
        self._l = 0.05

        self._resolution = 135

        self._tolerance = tolerance
        self._precision = precision
        self._tasks = tasks

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
        for rotateTask in self._tasks:
            dir_r_wheel = rotateTask.get_direction_of_right_wheel()
            dir_l_wheel = rotateTask.get_direction_of_left_wheel()
            target_radian = rotateTask.get_angle_rotation_radians()
            rate = rospy.Rate(self._precision)
            message = WheelsCmdStamped(vel_left=self._throttle*dir_l_wheel, vel_right=self._throttle*dir_r_wheel)
            self._distance_right = 0
            self._distance_left = 0
            while not rospy.is_shutdown():
                total_change_angle = ( self._distance_right - self._distance_left ) / (2*self._l)
                msg = f""" total_change_angle: {total_change_angle}, target_radian {target_radian}"""
                rospy.loginfo(msg)
                if abs(total_change_angle) >= (abs(target_radian) - self._tolerance):
                    stop = WheelsCmdStamped(vel_left=0, vel_right=0)
                    self._publisher.publish(stop)
                    break
                self._publisher.publish(message)
                rate.sleep()
            time.sleep(2)
        rospy.signal_shutdown(reason="tasks complete")
    
    def on_shutdown(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop)

class RotateTask():
    def __init__(self, radian):
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
        
if __name__ == "__main__":
    try:
        precision = 40 # published messages per second
        tolerance = 0.1 # accept total change in angle within tolerance
        throttle = 0.5
        # tasks = [RotateTask((-math.pi/2)), RotateTask((math.pi/2))]
        tasks = [RotateTask((math.pi/2))]
        node = Rotate(node_name="rotate_node", tasks=tasks, throttle=throttle, tolerance=tolerance, precision=precision)
        node.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass