import cv2
import numpy as np

class TagROperations:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TagROperations, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def getPerimeterInPixels(r):
        (ptA, ptB, ptC, ptD) = r.corners
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))

        tag_perimeter = (
            np.linalg.norm(np.array(ptA) - np.array(ptB)) +
            np.linalg.norm(np.array(ptB) - np.array(ptC)) +
            np.linalg.norm(np.array(ptC) - np.array(ptD)) +
            np.linalg.norm(np.array(ptD) - np.array(ptA))
        )

        return tag_perimeter