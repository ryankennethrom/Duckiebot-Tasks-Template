#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from custom_utils.constants import Stall
from custom_utils.final_tasks import *
import argparse

class FinalBehaviorMain(DTROS):
    def __init__(self, node_name, tasks):
        super(FinalBehaviorMain, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._tasks = tasks

        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._wheels_publisher = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)

    def run(self):
        for task in self._tasks:

            if not isinstance(task,FinalBehaviorMainTask):
                raise ValueError("task not recognized")

            task.execute(self)
            
        rospy.signal_shutdown(reason="tasks complete")

    def on_shutdown(self):
        stop = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._wheels_publisher.publish(stop)

if __name__ == "__main__":
    import math

    stall = Stall.TWO

    tasks = [
        # # part 1
        Tailing(),

        # part 2
        LeftRightTagTask(),

        # part 3
        RightLaneUntilCrosswalk(),
        WhiteLaneUntilCrossWalk(),
        Stop(stop_time=0),
        FreezeUntilDucksPass(),
        Stop(stop_time=3),
        LaneFollowUntilTimeout(base_velocity=0.25, timeout=3),
        Stop(stop_time=0),
        FindBrokenBot(detection_threshold=800),
        Stop(stop_time=3),
        SwitchLanesUntilSafe(base_velocity=0.25, detection_threshold=1000),
        Stop(stop_time=1),
        RightLaneUntilCrosswalk(),
        WhiteLaneUntilCrossWalk(),
        Stop(stop_time=0),
        FreezeUntilDucksPass(),
        Stop(stop_time=3),
        LaneFollowUntilIntersection(base_velocity=0.25),
        Stop(stop_time=3),

        # part 4
        TurnLeftTask(radians=math.pi/10, R=0, angular_velocity=3),
        StallAlignmentTask(target_stall=stall, proportional_gain=0.05, derivative_gain=0.05, integral_gain=0, velocity=0.35, integral_saturation=100),
        TagAlignmentTask(stall=stall, proportional_gain=0.007, derivative_gain=0.05, integral_gain=0, integral_saturation=0, debug=True, detection_sensitivity=300),
        ForwardParkingTask(target_stall=stall),
    ]

    node = FinalBehaviorMain(node_name="final_behavior_main_node", tasks=tasks)
    node.run()
    rospy.spin()