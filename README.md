
### Improvement from prior submission
To improve the few frames where the car deviated from the lane lines, the following modifications were done to the code
 1) Polynomial fit coefficients were averaged over the previous 5 frames 
 2) Frames were verified to check if there is significant change in lane width than the previous frame. If significant change occurred, then the current frame was dropped and the previous good frame was used instead.
 3) Slight tweaks were made to the color thresholds especially for the magnitude and binary thresholds. This resulted in a much cleaner binary thresholded image. 
 4) The source rectangle for perspective transform was tweaked to include a bit more distance on x-axis at the top portion of the rectangle. This gave a much cleaner and straighter warped image.
 
 Code changes are implemented in module `gen_linefit.py()`. Shown below is error correction if the change in lane width is more than ~15% from frame to frame or if the radius is very bad indicating a bad fit

```python
       if ((curr_road_width < 0.85*avg_road_width) | (curr_road_width > 1.15*avg_road_width) | (rc_rad < 130)):
            # cause for concern that there may be a gross error - ignore the frame
            print ("corrected %d %.1f %.1f %.1f"%(frame_no, curr_road_width, avg_road_width, rc_rad))        
            curr_road_width = avg_road_width
        else:
            avg_road_width = curr_road_width
            
            # if good fit, then add to the queue
            leftfit_q.append(left_fit)
            rightfit_q.append(right_fit)
            leftfit_q.pop(0)
            rightfit_q.pop(0)
```

Code below in module `gen_linefit.py()` implements averaging of coefficients over the last 5 frames. Those averaged coefficients are then used to fit the polynomial

```python
    # average last 5 frames and use those coefficients to fit the polynomial
    left_fit_avg = np.average(leftfit_q,axis =0)
    right_fit_avg = np.average(rightfit_q,axis =0)       

    left_fitx = left_fit_avg[0]*ploty**2 + left_fit_avg[1]*ploty + left_fit_avg[2]
    right_fitx = right_fit_avg[0]*ploty**2 + right_fit_avg[1]*ploty + right_fit_avg[2]    
```    

Other color spaces were experimented with instead of just the s-channle. My observation was that using only one channel is not optimal, rather using multiple color spaces in parallel and identifying the best fit to be used will lead to a more robust solution. Experiments show two threshold detectos of RV and SV channels can be used to improve the solution. 

Here is the modified [video output](./output_video/adv_lane_track_mod_2.mp4). As you can see the variations are much more gradual and the lane detection is improved.


# SDC Advanced Lane Detection
The goal of this project is to understand and implement a simple lane detection pipeline from the images recorded by a center car camera. Distortion induced by the camera is taken into account and correction is applied to the image feed. As described below, an image pipeline comprising of multiple transforms(color, gradient and perspective) is implemented resulting in a "birds-eye view" of the front facing camera image. Polynomial fits and various statistics during the process are captured and displayed.

From the output video, it is clear that simple image processing techniques are not sufficient to build a robust pipeline to detect lanes. Shadows on the road along with various road colors have a huge impact on detection accuracy. It will be interesting to further understand the accuracy of the current state-of-art approaches to lane detection. 

Key steps of this pipeline are:
* Calibrate and correct camera distortion using chessboard images 
* Generate calibration matrix and distortion coefficients and undistort images
* Apply color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to generate ("birds-eye view").
* Detect lane pixels and polyfit them to find the lane lines and warp back to the original image
* Anotate images with lane boundaries and metrics like lane curvature and offset from center

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---
### Code

Run the python notebook `lane_line_detection.ipynb` for the pipeline and lane detection video. Implementation consists of the following files located in the source directory

* source/lane_line_detection.ipynb  :   Capture video frames and runs image pipeline   
* source/gen_cam_cal.py             :   Generates camera calibration matrix and distortion coefficients using chessboard images
* source/gen_process_image.py       -   Implements Sobel gradients, color and perspective tranforms
* source/gen_linefit.py             -   Curve fit for lane detection  
* source/gen_stats_display.py       -   Calculates curvature, offset and implements polyfill and anotation of images
* source/gen_detection_pipeline.py  -   Implements the entire pipeline
* out_images                        -   Folder with images from various stages of the pipeline
* out_videos                        -   Folder with lane detected output videos 

### Camera Calibration

Camera matrix and distortion coefficients are calculated using a set of chessboard images and Opencv functions.

First step is to map 3D real world object points to 2D image space for the chessboard images. The chessboard images are fixed on a constant XY plane during capture, so object points are 3D points with Z axis of 0. 

```python
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
```

Then, we find internal corners of the chessboard using the Opencv function `cv2.findChessboardCorners()` and add the (x,y) coordinates to image space as shown below  
```python
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
```

Finally calibration matrix (mtx) and distortion coefficients (dst) are calculated using the `cv2.calibrateCamera()` function
```python
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```    

To remove distortion in an image, the function `cv2.undistort()` is applied with calibration matrix and distortion coefficients found above.
```python
    dst = cv2.undistort(img, cam_mtx, cam_dist, None, cam_mtx)
```

Applying this on chessboard images, we get 

![Original Distorted Image](./writeup_images/camera_dist_correct.png)

We can clearly see distortion at the top of the left image corrected after applying `cv2.undistort()`

### Image Pipeline 
#### 1. Distortion correction

Applying the same distortion correction as above
![alt text](./writeup_images/dist_road.png)

#### 2. Binary thresholding using Gradient and Color tranforms 

A combination of color and gradient thresholds was used to generate the binary image. Four different thresholds were used to generate the thresholded binary image. 

* S-color tranform
* SobelX gradient
* Sobel gradient magnitude
* Sobel gradient direction

The following thresholds were narrowed based on experimentation.

| Transform               | Threshold     | 
|:-----------------------:|:-------------:| 
| S color                 | 170, 255      | 
| SobelX grad             | 20, 100       |
| Sobel gradmagnitude     | 20, 100       |
| Sobel graddirection     | 0.7, 1.3      |

The final thresholded image is obtained by combining the various transforms as shown below. The code for thresholding is implemented in the file `source/gen_process_image.py`

```python
    combined_binary[(s_binary == 1) | (sxbinary == 1) | ((smagbinary == 1) & (sdirbinary == 1))] = 1
```

The images below show the effect of thresholding. The top image shows SobelX gradient and Color transform apllied, whereas the bottom image shows the result with all four thresholds applied

![alt text](./writeup_images/gradient_threshold.png)

#### 3. Perspective transform

The thresholded image is then run through a Perspective tranform to generate a birds-eye view image. This is accomplished by the opencv functions `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()`

```python 
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
```

This source and destination points taken for the perspective transform are shown below.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 100, 0        | 
| 735, 460      | 1180, 0       |
| 0, 720        | 100, 720      |
| 1280, 720     | 1180, 0       |

As expected the source and destination points we pick impact the tranformed image quite a bit. This is more pronounced when the images contain shadows. An interesting observation is that occasionally better perspective transform and lane detection are achieved when the source images were taken to the ends of the image (rather than to the ends of the lane). 

Shown below are a thresholded image before and after the perspective transform is applied 

![alt text](./writeup_images/perspective.png)


#### 4. Identifying lane-lines and polyfit

The next step is to identify lane lines from the perspective trasformed image. For most instances, thresolding coupled with perspective transform provide reasonably clean outlines of the lane pixels. A sliding window technique is then used to identify the lane pixels. 

This section is implemented in `gen_lanefit.py`

First, a histogram of ON pixels is run the bottom half of image. 

```python
    histogram = np.sum(warped_img[warped_img.shape[0]/2:,:], axis=0)
```

Then the location high intensity areas on the left and right sections of image are identified to give a starting location for the sliding window. 

```python
    end_margin_px = 100

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[end_margin_px:midpoint]) + end_margin_px
    rightx_base = np.argmax(histogram[midpoint+end_margin_px:histogram.shape[0]-100]) + midpoint + end_margin_px
```

The sliding window is moved along the the image and for each iteration of the window non-zero pixels in x and y direction are idenitifed.

```python
     good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
     good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
```

These good indices are appended to an array. At the end of each iteration, the mean of non-zero pixels is used to center the sliding windows of the next iteration. If there are not enough pixels, then the location of the window stays the same as before. 

```python
    if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
```

Once the sliding window is moved across the entire image, the non-zero x and y pixels are curve fitted using a 2nd order polynomial to detect lane lines  

```python
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
```

Shown below is the curve fitted lane lines with sliding windows and histogram of pixels  

![alt text](./writeup_images/curvefit.png)

Even in the limited test video provided, there are interesting cases where the entire thresholding and lane detection pipeline fails. They fall primarily in two areas
* Frames where the ends of the image do not have any active(ON) pixels since the line is dotted. Due to the nature of polyfit, this almost always returns an erroroneus fit
* Frames with shadows which make the processed images extremely noisy making it harder to even detect lane lines resulting in gross failures

Error correction for both these cases are implemented as shown below
In both these cases, the result is manifested as the right dotted white line detected being too far off (to the left or right) from its actual location. Here we measure the average road width and compare if it changed significantly (more than 15%) and apply correction  

First we measure the average roadwidth and curvature of the road as shown below

```python
    curr_road_width = np.average(right_fitx - left_fitx)    
    lc_rad, rc_rad = calc_curv(left_fitx, right_fitx, ploty)
```

If the detected roadwidth changes significantly compared to the previous frame, it is ignored. If the leftlane is calculated with good precision, then the right lane is calculated by just adding the average roadwidth to the left lane

```python
    if ((curr_road_width < 0.85*avg_road_width) | (curr_road_width > 1.15*avg_road_width) | (rc_rad < 50)):
         curr_road_width = avg_road_width
         right_fitx = left_fitx + curr_road_width
    else:
         avg_road_width = curr_road_width
```

#### 5. Metrics - Radius of curvature & Offset from center

Radius of curvature and vehicle offset from center is calculated in the file `gen_stats_display.py`

 First, the lanes detected in pixels are converted to lanes in real world meters and curve fitted 
```python
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
```    

and then radii of curvature are calculated based on the formula below

```python 
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix 
                                + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix 
                                + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

Offset from center is calculated based on the assumption that the camera is the center of the image. 
```python
    xm_per_pix = 3.7/700
    
    offset_px = (center - 0.5*(leftx[y_eval] + rightx[y_eval]))   
    offset = xm_per_pix * offset_px
```
    
#### 6. Pipeline output
All the functions for polyfill `filled_image()` and anotation `anotate_image()` are included in the file `gen_stats_display.py`

First the detected lane is mapped on the warped image using the function `cv2.fillPoly()` and it is then converted into original image space using inverse perspective transform `cv2.warpPerspective()`

```python
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (orig_image.shape[1], orig_image.shape[0]))     
```

This entire pipeline is implemented in the file `gen_detection_pipeline.py`. Shown below is an image before and after passing through the pipeline

![alt text](./writeup_images/pipeline.png)
---

### Video Output

Here are links to the [video output](./output_video/adv_lane_track.mp4).

Another version is shown [here](./output_video/adv_lane_track1.mp4). The difference in both videos is mostly due to the areas selected for perspective transform and thresholds selected for color and gradient transforms. 

---

### Discussion and further work
This project is aN introduction to camera calibration, color and perspective transforms and curve fitting functions. However, it is not very robust and depends heavily on many factors going right. 

As you can see the pipeline is not robust in areas where the road has strong shadows and is wobbly. Also sections of the road with lighter color(concrete sections) combined with reflections of the sun make detecting lane especially the white dotted right lines much harder. There is already significant volume of academic research on shadow detection and elimination in images and this is an area that i would like understand and implement in the near future.
