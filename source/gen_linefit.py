# Modules
# poly_fit_rimage() : Detects lane lines in an  image.
# bin_threshold_pipeline() : Use color & gradients transforms to create a thresholded binary image.
# warp_image : Apply a perspective transform to rectify binary image to generate a birds-eye view.

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from gen_stats_display import *


# ## Detect lanes and curve fit
#     1) Detect using histogram to initiate search and then windowing 
#     2) Curve Fit
#     3) Error checking and correction
#         * Check if road width is within expected range; apply correction if not
#         * Apply correction if 3 windows from either end of the image are empty, polyfit does not do a good job 
# Input : 
#    warped_img : birds-eye view image - perspective transform applied
#    frame_no : number of frame 
#    avg_road_width: expected avg lane width for error checking 
#
# Output:  
#     out_img - image with left and right line fit 
#     combined_binary - thresholded binary image with color, Sobel X gradient, mag and direction applied
#     left_fitx - X cordinates of left lane after polyfit 
#     right_fitx - X cordinates of right lane after polyfit 
#     ploty - Y coordinates  
#     adjusted - boolean - True if error correction applied on the image
#
def poly_fit_rimage(warped_img, frame_no, avg_road_width, leftfit_q, rightfit_q):
 
    # Choose the number of sliding windows
    nwindows = 9
   
    # Set height of windows
    window_height = np.int(warped_img.shape[0]/nwindows)
    
    histogram = np.sum(warped_img[int(warped_img.shape[0]/2):,:], axis=0)
    out_img = np.dstack((warped_img, warped_img, warped_img))*255
    
    end_margin_px = 100
    #Dont start search for the entire image, look within the perspective window to avoid corner cases
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[end_margin_px:midpoint]) + end_margin_px
    rightx_base = np.argmax(histogram[midpoint+end_margin_px:histogram.shape[0]-100]) + midpoint + end_margin_px
    
    #plt.plot(histogram)
    #print(midpoint, leftx_base, rightx_base)
    
    
    #print('window height',window_height)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    #print(nonzeroy.shape, nonzerox.shape, nonzeroy[1:100], nonzerox[1:100])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 100
    
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_img.shape[0] - (window+1)*window_height
        win_y_high = warped_img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
    
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
    
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    
        #print('good_left_inds', win_y_low, win_y_high, good_left_inds.shape, good_right_inds.shape, good_left_inds[1:10])
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
    
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                
    #if atleast two edge windows are zero on one side and none on the other, then use the curvature on the other side     
            
    # Check for the case if there are no elements in the edge windows
    curv_flag_l = 0
    curv_flag_r = 0
    samp_limit = 600
    adjusted = False
        
    a = np.array([len(left_lane_inds[in_val]) for in_val in range(0,nwindows)])
    b = np.array([len(right_lane_inds[in_val]) for in_val in range(0,nwindows)])
        
    if(all(a[:3] < samp_limit) or all(a[-3:] < samp_limit)): 
        curv_flag_l = 1 
        
    if(all(b[:3] < samp_limit) or all(b[-3:] < samp_limit)): 
        curv_flag_r = 1 
        
    left_good_window = np.argmax(a)
    right_good_window = np.argmax(b)
    ###
    
    #print(a, b, a[:3], a[-3:], curv_flag_l, curv_flag_r, left_good_window, right_good_window)
    #print(len(left_lane_inds),len(right_lane_inds))
        
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
        
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    #print(len(lefty), len(leftx), lefty[0:10], leftx[0:10])
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_fit_straight = np.polyfit(lefty, leftx, 1)
    right_fit_straight = np.polyfit(righty, rightx, 1)
    
    if (curv_flag_l == 1 and curv_flag_r == 0):
        valy = int(window_height*(nwindows - left_good_window - 0.5))
        valx = left_fit_straight[0]*valy + left_fit_straight[1]
        left_fit_intercept = valx - (right_fit[0]*valy**2 + right_fit[1]*valy)
        #print("in left", frame_no, valy, valx, left_fit_intercept)
    
    if (curv_flag_r == 1 and curv_flag_l == 0):
        valy = int(window_height*(nwindows - right_good_window - 0.5))
        valx = right_fit_straight[0]*valy + right_fit_straight[1]
        right_fit_intercept = valx - (left_fit[0]*valy**2 + left_fit[1]*valy)
        #print("in right", frame_no, valy, valx, right_fit_intercept)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0] )
    
    #if (curv_flag_l == 1 and curv_flag_r == 0):
    #    left_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + left_fit_intercept
    #    adjusted = True
    #    #left_fitx = left_fit_straight[0]*ploty + left_fit_straight[1]
    #else:
    #    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    
        
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
    ## additional check to eliminate error frames
    ## Check roadwidth to ensure that it is within 20% of expected width
    curr_road_width = np.average(right_fitx - left_fitx)
    
    lc_rad, rc_rad = calc_curv(left_fitx, right_fitx, ploty)

    if (frame_no < 6):
        avg_road_width = curr_road_width
        leftfit_q.append(left_fit)
        rightfit_q.append(right_fit)
    
    else:
        if ((curr_road_width < 0.85*avg_road_width) | (curr_road_width > 1.15*avg_road_width) | (rc_rad < 130)):
            # cause for concern that there may be a gross error, check for curvatures for further proof
            #print ("corrected %d %.1f %.1f %.1f"%(frame_no, curr_road_width, avg_road_width, rc_rad))        
            curr_road_width = avg_road_width
            #right_fitx = left_fitx + curr_road_width            
        
        else:
            avg_road_width = curr_road_width
            leftfit_q.append(left_fit)
            rightfit_q.append(right_fit)
            leftfit_q.pop(0)
            rightfit_q.pop(0)
    
    left_fit_avg = np.average(leftfit_q,axis =0)
    right_fit_avg = np.average(rightfit_q,axis =0)       

    left_fitx = left_fit_avg[0]*ploty**2 + left_fit_avg[1]*ploty + left_fit_avg[2]
    right_fitx = right_fit_avg[0]*ploty**2 + right_fit_avg[1]*ploty + right_fit_avg[2]
    
    #print("out", frame_no, left_fit[0], right_fit[0], ";", left_fit[2], right_fit[2], "; %.1f %.1f"%(avg_road_width, rc_rad))

    #print("fit queue",len(leftfit_q), len(rightfit_q), leftfit_q, rightfit_q)    

    return out_img, left_fitx, right_fitx, ploty, adjusted, curr_road_width, leftfit_q, rightfit_q
