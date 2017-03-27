# Modules
# img_pipleine() : Full lane detection pipeline
# 
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import gen_process_image
import gen_stats_display
import gen_linefit


## Input
#     image : Calibrated image
#     mtx : Camera calibration matrix
#     dist : Camera distortion coefficients
#     source_rect : 4x2 numpy array source rectangle selection for warping
#     dest_rect : 4x2 numpy array destination area selection for warping
#     frame_no : frame no for anotation 
#     avg_road_width : estimated average lane width for error correction purpose only
#  Output
#     result_img : lane filled and anotated image
#     curr_road_width : Averge lane width of this frame for error correction purposes only
#     
def img_pipeline(image, mtx, dist, src_rect, dest_rect, frame_no, avg_road_width, leftfit_q, rightfit_q):    
    
    dst = gen_process_image.undistort_image(image, mtx, dist);
    
    result_color_bin, result_combined_bin = gen_process_image.bin_threshold_pipeline(dst);
    warped_img, Minv = gen_process_image.warp(result_combined_bin, src_rect, dest_rect)
    
    polyfit_img, left_fitx, right_fitx, ploty, adjust_bool, curr_road_width, lf_q, rf_q = gen_linefit.poly_fit_rimage(warped_img, frame_no, avg_road_width, leftfit_q, rightfit_q)
    
    left_curverad, right_curverad = gen_stats_display.calc_curv(left_fitx, right_fitx, ploty)
    center_offset = gen_stats_display.calc_offset(polyfit_img, left_fitx, right_fitx, ploty)
    
    filled_img, pts = gen_stats_display.filled_image(image, warped_img, Minv, left_fitx, right_fitx, ploty)
    
    result_img = gen_stats_display.anotate_image(filled_img, left_curverad, right_curverad, center_offset, frame_no)
    
    #print("No: %d, lc:%.1f rc:%.1f crw:%.1f offs:%.3f"%(frame_no, left_curverad, right_curverad, curr_road_width, center_offset))

    return result_img, curr_road_width, left_fitx, right_fitx, lf_q, rf_q


