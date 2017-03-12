# SDC Advanced Lane Detection
The goal of this project is to understand and implement a simple lane detection pipeline from the images recorded by a center car camera. Distortion induced by the camera is taken into account and correction is applied to the image feed. As described below, an image pipeline comprising of multiple transforms is implemented resulting in a "birds-eye view" of the front facing camera image. Polynomial fits and various statistics during the process are captured and displayed.

It is clear that simple image processing techniques are not sufficient to build a robust pipeline to detect lanes. Shadows on the road along with various road colors have a huge impact on detection accuracy. It will be interesting to further understand the accuracy of the current state-of-art approaches to lane detection. 

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---
###Writeup / README

You're reading it!

Run the python notebook for the pipeline and lane detection video. Implementation consists of the following files located in the source directory

* lane_line_detection.ipynb  - Runs the pipeline on individual test images and  video  
* gen_cam_cal.py             - Generates camera calibration matrix and distortion coefficients using chessboard images
* gen_process_image.py       - Implements Sobel gradients, color and perspective tranforms
* gen_linefit.py             - Curve fits for lane detection  
* gen_stats_display.py       - Calculates curvature, offset and implemets polyfill and anotation of images
* gen_detection_pipeline.py  - Implemets the entire pipeline
    
###Camera Calibration

Camera matrix and distortion coefficients are calculated using a set of chessboard images and Opencv functions.

First step is to map 3D real world object points to 2D image space for the chessboard images. The chessboard images are fixed on a constant XY plane during the capture of chessboard images, so object points are 3D points with Z axis of 0. 

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

###Pipeline (single images)

####1. Distortion correction

Applying the same distortion correction as above
![alt text](./writeup_images/dist_road.png)

####2. Binary thresholding using Gradient and Color tranforms 

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

The below images show the effect of thresholding. The top image shows SobelX gradient and Color transform apllied, whereas the bottom image shows the result with all four thresholds applied

![alt text](./writeup_images/gradient_threshold.png)

####3. Perspective transform

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

As expected the source and destination points we pick impact the tranformed image quite a bit. This is more pronounced when the images contain shadows. An interesting observation is the occasionally better perspective transform and lane detection was achieved when the source images were taken to the ends of the image, rather than to the ends of the lane. 

Shown below are a thresholded image before and after the perspective transform is applied 

![alt text](./writeup_images/perspective.png)


####4. Identifying lane-lines and polyfit

The next step is to identify lane lines from the perspective trasformed image. For most instances, thresolding coupled with perspective transform provide reasonably clean outlines of the lane pixels. A sliding window technique is then used to identify the lane pixels. 

This section is implemented in `gen_lanefit.py`

First, a histogram of ON pixels is run the bottom half of image. 
```python
    histogram = np.sum(warped_img[warped_img.shape[0]/2:,:], axis=0)
```
Then the location high intensity areas on the left and right sections of image are identified to give a starting location for the sliding window. 

```python
    end_margin_px = 100
    #Dont start search for the entire image, look within the perspective window to avoid corner cases
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[end_margin_px:midpoint]) + end_margin_px
    rightx_base = np.argmax(histogram[midpoint+end_margin_px:histogram.shape[0]-100]) + midpoint + end_margin_px
```

The sliding window is moved along the the image and for each iteration of the window non-zero pixels in x and y direction are idenitifed.

```python
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
```
These good indices are appended to an array. At the end of each iteration, the mean of non-zero pixels is used to center the sliding windows of the next iteration. If there are not enough pixels, then the location of the window stays the same as before. 

```python
    if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
```python

Once the sliding window is moved across the entire image, the non-zero x and y pixels are curve fitted using a 2nd order polynomial to detect lane lines  

```python
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
```

Shown below is the curve fitted lane lines with sliding windows and histogram of pixels  

![alt text](./writeup_images/curvefit.png)

Even in the limited test video provided, there are interesting cases where the entire thresholding and lane detection pipeline fails. They fall primarily in two areas
1) Frames where the ends of the image do not have any active(ON) pixels since the line is dotted. Due to the nature of polyfit, this almost always returns an erroroneus fit
2) Frames with shadows which make the processed images extremely noisy making it harder to even detect lane lines resulting in gross failures

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

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text](./writeup_images/pipeline.png)
---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video/adv_lane_track.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
