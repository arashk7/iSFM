import cv2
import numpy as np
from dataclasses import dataclass

from dataclasses import dataclass
import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class ShiTomasiAndORB:
    max_corners: int = 500
    quality_level: float = 0.01
    min_distance: int = 10

    image: np.ndarray = None
    corners: np.ndarray = None

    def set_image(self, img: np.ndarray):
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("Invalid image input. Must be a non-null numpy.ndarray.")
        self.image = img
        logging.info("Image set successfully.")

    def shi_tomasi_corner_detect(self) -> np.ndarray:
        if self.image is None:
            raise ValueError("Image not set. Use set_image() method to set the image first.")
        
        try:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.corners = cv2.goodFeaturesToTrack(gray_image, self.max_corners, self.quality_level, self.min_distance)
            logging.info("Shi-Tomasi corner detection completed.")
            return self.corners
        except Exception as e:
            logging.error(f"Error in Shi-Tomasi corner detection: {e}")
            raise

    def orb_descriptor(self) -> tuple:
        if self.corners is None:
            raise ValueError("Corners not set. Use shi_tomasi_corner_detect() method to detect corners first.")

        try:
            keypoints = [cv2.KeyPoint(x=corner[0][0], y=corner[0][1], size=20) for corner in self.corners]
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.compute(self.image, keypoints)
            logging.info("ORB descriptor computation completed.")
            return keypoints, descriptors
        except Exception as e:
            logging.error(f"Error in ORB descriptor computation: {e}")
            raise


a = ShiTomasiAndORB()


# Read the image
image = cv2.imread(r'DS\beethoven_data\images\0000.ppm')

a.set_image(image)
c = a.shi_tomasi_corner_detect()
k,d = a.orb_descriptor()


# Draw the keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, k, None, color=(0, 255, 0), flags=0)

# Display the image with keypoints
cv2.imshow('Keypoints and Descriptors', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
