from enum import Enum
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String
import os
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA

class Colors(Enum):
    Red = [1.0, 0.0, 0.0]
    Green = [0.0, 1.0, 0.0]
    Blue = [0.0, 0.0, 1.0]
    Yellow = [1.0, 1.0, 0.0]
    Teal = [0.0, 1.0, 1.0]
    Magenta = [1.0, 0.0, 1.0]
    Off = [0.0, 0.0, 0.0]
    DarkOrange = [1.0, 0.55, 0]
    White = [1.0, 1.0, 1.0]  # White color

class ColorOperations:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageOperations, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def getLedMessage(colorPattern):
        led_msg = LEDPattern()

        for color in colorPattern.getColorMask():
            # Color for the LEDs
            rgba = ColorRGBA()
            rgba.r = color[0]
            rgba.g = color[1]
            rgba.b = color[2]
            rgba.a = 1.0

            led_msg.rgb_vals.append(rgba)
        
        return led_msg
    
class ColorPattern():
    def __init__(self, frontLeft, frontRight, backLeft, backRight):
        if (not isinstance(frontLeft, Colors) or 
            not isinstance(frontRight, Colors) or
            not isinstance(backLeft, Colors) or 
            not isinstance(backRight, Colors)):
            raise Exception("Parameters of ColorPattern must be of type Colors(Enum)")
        self.frontLeft = frontLeft
        self.frontRight = frontRight
        self.backLeft = backLeft
        self.backRight = backRight

    def getColorMask(self):
        return [self.frontLeft.value, self.backRight.value, [0,0,0],  self.backLeft.value, self.frontRight.value]