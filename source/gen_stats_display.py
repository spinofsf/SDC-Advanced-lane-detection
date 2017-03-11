# Modules
# calc_curve() : Calculate curvature of left and line fits
# calc_offset() : Calculates offset from the center of the image
# filled_image() : Applies a poly fill to show the detected lane
# anotate_image() : Anotates an image with stats/metrics 
# curv_check() : Checks if the curvature of left and right lanes are off significantly which indicates a detection error

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

## Calculate curvature of the left and right lanes
def calc_curv(leftx,rightx, ploty):
    
    y_eval = np.max(ploty)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

## Calculate offset from center of the image 
def calc_offset(image, leftx, rightx, ploty):

    y_eval = int(np.max(ploty))
    
    center = 0.5* image.shape[1]
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    #Calculate offset from center
    offset_px = (center - 0.5*(leftx[y_eval] + rightx[y_eval]))   
    offset = xm_per_pix * offset_px
    
    #print(frame_no, y_eval, leftx[y_eval], rightx[y_eval], offset_px, offset)
    
    return offset


## Take an image and polyfills the detected lane 
def filled_image(orig_image, warp_image, Minv, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warp_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (orig_image.shape[1], orig_image.shape[0])) 
    
    # Combine the result with the original image
    result_img = cv2.addWeighted(orig_image, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    
    return result_img


## Anotates an image with a statistics label of curvature and offset 
def anotate_image(ant_image, lcurve, rcurve, offset, frame_no):

    label_str = 'Frame:%d curvature:%.1fm offset:%.1fm' %(frame_no, lcurve, offset)
    ant_img = cv2.putText(ant_image, label_str, (560,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)

    return ant_img


## Checks if curvature of left and right lanes are off significantly (more than 8x) 
def curv_check(lcurve, rcurve):
    
    curve_ratio = lcurve/rcurve
    
    if (curve_ratio < 0.0125 or curve_ratio > 8):
            curve_error = True
    else:
            curve_error = False
            
    return curve_error
