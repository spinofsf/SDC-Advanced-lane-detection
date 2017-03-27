# Modules
# undistort_image() : Apply a distortion correction to a single image.
# bin_threshold_pipeline() : Use color & gradients transforms to create a thresholded binary image.
# warp_image : Apply a perspective transform to rectify binary image to generate a birds-eye view.

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

## Undistort chessboard images using the distortion matrix
# 
# Input : 
#    img : source image that needs to be undistorted
#    cam_mtx : camera calibration matrix
#    cam_dst : camera distotion coefficients 
# Output:  
#     dst - undistorted image  
# 
def undistort_image(img, cam_mtx, cam_dist):
    
    dst = cv2.undistort(img, cam_mtx, cam_dist, None, cam_mtx)
    
    return dst


## Apply Binary thresholding on Sobel gradient, Sobel Mag and dir, and S-channel Color
# 
# Input : 
#    image : source image 
#    s_thresh : tuple of thresold low and high levels for s-color channel 
#    sx_thresh : tuple of thresold low and high levels for Sobel X gradient 
#    smag_thresh : tuple of thresold low and high levels for Sobel Magnitude 
#    sdir_thresh : tuple of thresold low and high levels for Sobel Direction  
# 
# Output:  
#     color_binary - thresholded binary image with color and sobelX thresholding only 
#     combined_binary - thresholded binary image with color, Sobel X gradient, mag and direction applied
# 
def bin_threshold_pipeline(image, s_thresh=(170, 255), sx_thresh=(20, 100), smag_thresh=(20,100), sdir_thresh=(0.7,1.3)):
   
    #s_thresh=(170, 255), sx_thresh=(20, 100), smag_thresh=(20,100), sdir_thresh=(0.7,1.3)

    img = np.copy(image)
    img = img[:,:,::-1]

    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
      
    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_x = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel_x)
    sxbinary[(scaled_sobel_x >= sx_thresh[0]) & (scaled_sobel_x <= sx_thresh[1])] = 1

    # Sobel y
    sobely = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1) # Take the derivative in x
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_y = np.uint8(255*abs_sobely/np.max(abs_sobely))

    # Sobel magnitude and direction
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel_mag = np.uint8(255*sobel_mag/np.max(sobel_mag))

    sobel_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # Sobel magnitude and direction threshold
    smagbinary = np.zeros_like(scaled_sobel_mag)
    smagbinary[(scaled_sobel_mag >= smag_thresh[0]) & (scaled_sobel_mag <= smag_thresh[1])] = 1

    sdirbinary = np.zeros_like(sobel_dir)
    sdirbinary[(sobel_dir >= sdir_thresh[0]) & (sobel_dir <= sdir_thresh[1])] = 1

    # Threshold y gradient
    sybinary = np.zeros_like(scaled_sobel_y)
    sybinary[(scaled_sobel_y >= sx_thresh[0]) & (scaled_sobel_y <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | ((sxbinary == 1) & (sybinary==1)) | ((smagbinary == 1) & (sdirbinary == 1))] = 1

    return color_binary, combined_binary


## Apply Perspective transform
# 
# Input : 
#    image : source image 
#    src_rect: 4x2 array - coordinates of rectangle on the source image on which the transform is applied  
#    dst_rect: 4x2 array - coordinates of rectangle on the warped image corresponding to src_rect 
# Output:  
#    warped : image with perspective transform applied
#    Minv : coefficients for inverse perspective transform 
# 
def warp(image, src_rect, dst_rect):
    
    image_size = (image.shape[1], image.shape[0])
    
    M = cv2.getPerspectiveTransform(src_rect, dst_rect)
    Minv = cv2.getPerspectiveTransform(dst_rect, src_rect)
    warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
    
    return warped, Minv
