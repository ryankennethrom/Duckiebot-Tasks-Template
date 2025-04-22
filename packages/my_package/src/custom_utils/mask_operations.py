import numpy as np 
import math

class MaskOperations:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MaskOperations, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def getActiveCount(mask):
        return np.count_nonzero(mask)
    
    @staticmethod
    def getActiveCenter(mask):
        if mask is None:
            return (-math.inf, -math.inf)

        y_coords, x_coords = np.where(mask > 0)

        if len(x_coords) == 0 or len(y_coords) == 0:
            return (-math.inf, -math.inf)

        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))

        return (center_x, center_y)
    
    @staticmethod
    def keep_middle_half(mask):
        height, width = mask.shape
        start = width // 4
        end = 3 * width // 4
        cropped_mask = np.zeros_like(mask)
        cropped_mask[:, start:end] = mask[:, start:end]
        return cropped_mask
        
    @staticmethod
    def getActiveCenterX(mask):
        if mask is None:
            return -math.inf
        
        _, x_coords = np.where(mask > 0)

        if len(x_coords) == 0:
            return -math.inf
        
        return int(np.mean(x_coords))
    
    @staticmethod
    def getLowestY(mask):
        if mask is None:
            return -math.inf

        y_coords = np.where(mask > 0)[0]

        if len(y_coords) == 0:
            return -math.inf

        lowest_y = np.max(y_coords)

        return int(lowest_y)
    
    @staticmethod
    def getMaskVarianceAxisX(mask):
        _, x_coords = np.where(mask > 0)

        if len(x_coords) == 0:
            return math.inf
        
        x_variance = np.var(x_coords)

        return x_variance

    @staticmethod
    def computeErrorInAxisX(mask, target_x, pixel_value):
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) == 0:
            return 0

        errors = (x_coords - target_x) * pixel_value

        return np.sum(errors)