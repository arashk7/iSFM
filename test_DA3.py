import os
import cv2
import numpy as np
from dataclasses import dataclass
import logging
from ak_DataAssociation import ShiTomasiAndORB, BruteForceMatcher


# this is your dataset path
ds_path = r'DS\diamond_walk\diamond_walk\cam0\data'
K=[600,   0,   320,
    0,     600, 240,
    0,     0,   1]
K=np.reshape(K,(3,3))

sao = {}

bf = BruteForceMatcher()

def recoverpos(E):
    # Singular Value Decomposition (SVD) of the essential matrix
    U, _, Vt = np.linalg.svd(E)

    # Ensure that the determinant of U and Vt is positive to make them rotation matrices
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # Compute the rotation matrix R from U and Vt
    R = np.dot(U, Vt)

    # Compute the translation vector t
    t = U[:, 2]

    # Ensure that t is a unit vector
    t /= np.linalg.norm(t)
    return R,t
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

        image_with_keypoints = a.draw_keypoints()

        if index > 0:
            diff = np.int64(a.descriptors[0]) - np.int64(sao[index-1].descriptors[0])
            

        
            # FLANN parameters for feature matching
            # FLANN_INDEX_KDTREE = 1
            # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) 
            # search_params = dict(checks=50)

            # flann = cv2.FlannBasedMatcher(index_params, search_params)

            descriptors1 = sao[index-1].descriptors
            descriptors2 = a.descriptors

            keypoints1 = sao[index-1].keypoints
            keypoints2 = a.keypoints
            
            #print(keypoints1)
            # Brute-Force Matcher
            
            matches = bf.match(descriptors1,descriptors2)


            # Filter for good matches using Lowe's ratio test
            good_matches = []
            rate = 1
            for m, n in matches:
                if m.distance < rate * n.distance:
                    good_matches.append(m)
                    
            if len(good_matches)>8:
                pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

                F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC )

                if F is None or F.shape != (3, 3):
                    print("Error in computing the fundamental matrix")
                else:
                    print("Fundamental Matrix:\n", F)
             

            # Filter the outlier matches
            inlier_pts1 = pts1[mask.ravel() == 1]
            inlier_pts2 = pts2[mask.ravel() == 1]

            inlier_keypoints1 = [cv2.KeyPoint(x=p[0], y=p[1], size=20) for p in inlier_pts1]
            inlier_keypoints2 = [cv2.KeyPoint(x=p[0], y=p[1], size=20) for p in inlier_pts2]
            # print('done')

            for pt1, pt2 in zip(inlier_pts1, inlier_pts2):
                x = np.array([pt1[0], pt1[1], 1])
                x_prime = np.array([pt2[0], pt2[1], 1])
                error = np.dot(np.dot(x_prime.T, F), x)
                # print("Epipolar Constraint Error:", error)

            image_with_keypoints = cv2.drawKeypoints(image_with_keypoints, inlier_keypoints2, None, color=(255, 255, 0), flags=0)

            E = np.dot(np.dot(K.T, F), K) 
            

            _, R, t, _ = cv2.recoverPose(E, inlier_pts1, inlier_pts2, K)
            R1,t1 = recoverpos(E)
            
            print(R)
            print(R1)
            print(t)
            print(t1)
            
        # Display the image with keypoints
        cv2.imshow('Keypoints and Descriptors', image_with_keypoints)
        cv2.waitKey(0)


cv2.destroyAllWindows()
