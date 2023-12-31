import os
import cv2
import numpy as np
from dataclasses import dataclass
import logging
from ak_DataAssociation import ShiTomasiAndORB


# this is your dataset path
ds_path = r'DS\beethoven_data\images'

sao = {}


# Loop through each file in the directory
for index, filename in enumerate(os.listdir(ds_path)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.ppm')):

        # Construct the full file path
        file_path = os.path.join(ds_path, filename)


        image = cv2.imread(file_path)


        a = ShiTomasiAndORB()

        a.set_image(image)

        a.run() 
        
        sao[index] = a

        if index > 0:
            diff = np.int64(a.descriptors[0]) - np.int64(sao[index-1].descriptors[0])
            print(diff)

        
            # FLANN parameters for feature matching
            # FLANN_INDEX_KDTREE = 1
            # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) 
            # search_params = dict(checks=50)

            # flann = cv2.FlannBasedMatcher(index_params, search_params)

            descriptors1 = sao[index-1].descriptors
            descriptors2 = a.descriptors
            if descriptors1 is None or descriptors2 is None:
                raise ValueError("One of the descriptor sets is empty.")
            if descriptors1.dtype != np.uint8 or descriptors2.dtype != np.uint8:
                raise ValueError("Descriptors data type must be uint8.")
            if len(descriptors1) == 0 or len(descriptors2) == 0:
                raise ValueError("One of the descriptor sets is empty.")

            # Brute-Force Matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(descriptors1,descriptors2, k=2)

            # Filter for good matches using Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
                    print('.')

        image_with_keypoints = a.draw_keypoints()

        # Display the image with keypoints
        cv2.imshow('Keypoints and Descriptors', image_with_keypoints)
        cv2.waitKey(0)


cv2.destroyAllWindows()
