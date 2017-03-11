# 
#  Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
#
# Input : List of chessboard images that are used for calibration
# Output: mtx - camera calibration matrix 
#         dst - distortion coefficients 
# 

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def cal_cam(images):
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    # images = glob.glob('../camera_cal/calibration*.jpg')

    #print('total images are :', images)
    #print(type(images), size(images))
    # Step through the list and search for chessboard corners
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #print(gray.shape, gray.shape[0], gray.shape[1])
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            cv2.imshow('img',img)
            cv2.waitKey(20)

    
    cv2.destroyAllWindows()
    
    # Calibrate camera and apply distortion matrix to test images
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return mtx,dist

