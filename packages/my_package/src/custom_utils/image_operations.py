import cv2
import numpy as np

class ImageOperations:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageOperations, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def undistort(raw_bot_image):
        # Camera intrinsics
        K = np.array([[310.0149584021843, 0.0, 307.7177249518777],
                    [0.0, 309.29643750324607, 229.191787718834],
                    [0.0, 0.0, 1.0]], dtype=np.float32)

        D = np.array([-0.2683225140828933, 0.049595473114203516,
                    0.0003617920649662741, 0.006030049583437601, 0.0], dtype=np.float32)

        img_width, img_height = 640, 480

        # Compute optimal camera matrix to minimize black areas after undistortion
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (img_width, img_height), 1, (img_width, img_height))

        # Undistort image
        undistorted = cv2.undistort(raw_bot_image, K, D, None, new_K)

        # Crop the image based on ROI (Region of Interest)
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

        h, w, _ = undistorted.shape

        # point = [120, 191]
        # print(point)
        # cv2.circle(undistorted, tuple(point), 5, (0, 0, 255), -1)  # Red dots

        return undistorted
    
    @staticmethod
    def getAnnotateImage(image, r):
        # Convert grayscale image to BGR so we can draw colors
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        (ptA, ptB, ptC, ptD) = r.corners
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))

        # Bounding box all in green
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)

        # Draw center in Cyan
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (255, 255, 0), -1)

        # Tag ID in Bright Purple
        cv2.putText(image, str(r.tag_id), (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 0, 204), 2)

        return image
    
    @staticmethod
    def getAprilDetectionResults(image, detector):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        results = detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        return results

    @staticmethod
    def getWhiteMask(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define HSV range for detecting **white color**
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 50, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        return mask_white

    @staticmethod
    def getYellowMask(image):
        hsv = cv2.cvtColor((image), cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        return mask_yellow
    
    @staticmethod
    def getBlueMask(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 0], dtype=np.uint8) 
        upper_blue = np.array([140, 255, 255], dtype=np.uint8)  
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        return mask_blue

    @staticmethod
    def getRedMask(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 100, 100], dtype=np.uint8)
        upper_red = np.array([10, 255, 255], dtype=np.uint8)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        return mask_red
    
    @staticmethod
    def getCenterAxisX(image):
        return image.shape[1] // 2
    
    @staticmethod
    def getImageWidth(image):
        return image.shape[1]
    
    @staticmethod
    def getCenterAxisY(image):
        return image.shape[0] // 2
    
    @staticmethod
    def getDuckiebotBlueMask(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 100, 50], dtype=np.uint8)  # Lower bound for dark blue
        upper_blue = np.array([120, 255, 100], dtype=np.uint8)  # Upper bound for dark blue
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        return mask_blue
    
    @staticmethod
    def getCrosswalkBlueMask(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 0], dtype=np.uint8)  # Lower bound for blue
        upper_blue = np.array([140, 255, 255], dtype=np.uint8)  # Upper bound for blue
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        return mask_blue
    
    @staticmethod
    def getMaskLines(mask):
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
        out = []
        if lines is not None:
            out.extend(lines[:, 0])
        return out
    
    @staticmethod
    def getImageLines(image):
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
        return lines
    
    @staticmethod
    def getHomography(undistorted_image):
        image = undistorted_image
        h, w, _ = image.shape

        img_size = (w, h)
        
        src = np.float32([
            [0,382],
            [224, 191],
            [589, 382],
            [364, 191],
        ])

        dst = np.float32([
            [100, 382],
            [100, 0],
            [489, 382],
            [489, 0],
        ])

        # cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)  # Red dots

        M = cv2.getPerspectiveTransform(src, dst)
        
        warped = cv2.warpPerspective(image, M, img_size)

        return warped
    
    @staticmethod
    def getGrayscale(image):
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayscale_image
    
    @staticmethod
    def getErrorFromCenterInAxisX(image, object_x):
        height, width = image.shape[:2]
        center_x = width // 2
        error_x = object_x - center_x
        return error_x