import cv2
import numpy as np
from dataclasses import dataclass
import logging
from ak_DataAssociation import ShiTomasiAndORB



image = cv2.imread(r'DS\beethoven_data\images\0000.ppm')


a = ShiTomasiAndORB()

a.set_image(image)

a.run()



image_with_keypoints = a.draw_keypoints()


# Display the image with keypoints
cv2.imshow('Keypoints and Descriptors', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
