import cv2
import numpy as np
from dataclasses import dataclass
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

    keypoints: np.ndarray = None
    descriptors: np.ndarray = None

    def set_image(self, img: np.ndarray):
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("Invalid image input. Must be a non-null numpy.ndarray.")
        self.image = img
        logging.info("Image set successfully.")

    def run(self):
        self.shi_tomasi_corner_detect()
        self.orb_descriptor()
        
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
            self.keypoints, self.descriptors = keypoints, descriptors
            return keypoints, descriptors
        except Exception as e:
            logging.error(f"Error in ORB descriptor computation: {e}")
            raise
    
    def draw_keypoints(self)-> np.ndarray:
        if self.keypoints is None:
            raise ValueError("Keypoints not set. Use shi_tomasi_corner_detect() then orb_descriptor methods to detect corners make the keypoints and descriptors first.")
        
        if self.image is None:
            raise ValueError("Image not set. Use set_image() method to set the image first.")
        
        try:
            image_with_keypoints = cv2.drawKeypoints(self.image, self.keypoints, None, color=(0, 255, 0), flags=0)
            logging.info(f"Keypoints has drawn successfully.")
            return image_with_keypoints
        except Exception as e:
            logging.error(f"Error in drawing keypoints: {e}")
            raise