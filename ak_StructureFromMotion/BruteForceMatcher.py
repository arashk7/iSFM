import cv2
import numpy as np
from dataclasses import dataclass
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class BruteForceMatcher:

    bf = None
    def __post_init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def match(self, descriptors1, descriptors2):

        if descriptors1 is None or descriptors2 is None:
            raise ValueError("One of the descriptor sets is empty.")
        if descriptors1.dtype != np.uint8 or descriptors2.dtype != np.uint8:
            raise ValueError("Descriptors data type must be uint8.")
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            raise ValueError("One of the descriptor sets is empty.")

        matches = self.bf.knnMatch(descriptors1,descriptors2, k=2)

        return matches
    