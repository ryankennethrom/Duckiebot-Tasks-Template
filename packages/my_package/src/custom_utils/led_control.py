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

class LEDState():
    def __init__(self, message_name, colorPattern):
        if(not isinstance(colorPattern, ColorPattern)):
            raise Exception("colorPattern must be of type ColorPattern")
        self.message_name = message_name
        self.led_colors = colorPattern.getColorMask()
    
    def getLedMessage(self):
        led_msg = LEDPattern()

        for color in self.led_colors:
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