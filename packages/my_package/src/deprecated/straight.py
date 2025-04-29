#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
import math


# constants
RESOLUTION = 135
THROTTLE = 0.8
DIRECTION = 1
TOLERANCE = 0.05

class Straight(DTROS):
    def __init__(self, node_name, tasks):
        super(Straight, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

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
        self._distance = 0

        self._tasks = tasks
        self._task_counter = 0

    def callback_left(self, data):
        if self._ticks_left is None:
            self._ticks_left = data.data
            return
        
        self._distance_left += 2*math.pi*self._radius*((data.data - self._ticks_left)/RESOLUTION)
        self._ticks_left = data.data
        self.update_distance()
    
    def callback_right(self, data):
        if self._ticks_right is None:
            self._ticks_right = data.data
            return
        
        self._distance_right += 2*math.pi*self._radius*((data.data - self._ticks_right)/RESOLUTION)
        self._ticks_right = data.data
        self.update_distance()

    def update_distance(self):
        self._distance = (self._distance_left + self._distance_right) / 2

    def run(self):
        # publish received tick messages every 0.05 second (20 Hz)
        rate = rospy.Rate(10)
        goal = self._distance + self._tasks[self._task_counter]["distance"] * self._tasks[self._task_counter]["direction"]
        
        while not rospy.is_shutdown():
            # if self._ticks_right is not None and self._ticks_left is not None:
            #     # start printing values when received from both encoders
            # msg = f"Distances and tick [LEFT, RIGHT]: {self._distance_left} : {self._ticks_left}, {self._distance_right} : {self._ticks_right}"
            lower_bound = goal - TOLERANCE
            upper_bound = goal + TOLERANCE
            msg = f"Distance: {self._distance}, Goal: {goal}"
            rospy.loginfo(msg)
            
            if (self._distance >= lower_bound and self._distance <= upper_bound) or self._distance == goal:
                stop = WheelsCmdStamped(vel_left=0, vel_right=0)
                self._publisher.publish(stop)
                self._task_counter += 1
                if self._task_counter >= len(self._tasks):
                    rospy.signal_shutdown(reason="tasks complete")
                else:
                    goal = self._distance + self._tasks[self._task_counter]["distance"] * self._tasks[self._task_counter]["direction"]
                
            else:
                message = WheelsCmdStamped(vel_left=THROTTLE*self._tasks[self._task_counter]["direction"], vel_right=THROTTLE*self._tasks[self._task_counter]["direction"])
                self._publisher.publish(message)

            rate.sleep()
    
    def on_shutdown(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop)

if __name__ == "__main__":
    tasks = [{"distance": 1.25, "direction": 1}, {"distance": 1.25, "direction": -1}]
    node = Straight(node_name="straight_node", tasks=tasks)
    node.run()
    rospy.spin()