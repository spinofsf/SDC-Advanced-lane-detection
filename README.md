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

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

Run the python notebook for the pipeline and lane detection video. Implementation consists of the following files. 

lane_line_detection.ipynb - Runs the pipeline on individual test images and  video  
gen_cam_cal.py - Generates camera calibration matrix and distortion coefficients using chessboard images
gen_process_image.py - Implements Sobel gradients, color and perspective tranforms
gen_linefit.py - Curve fits for lane detection  
gen_stats_display.py - Calculates curvature, offset and implemets polyfill and anotation of images
gen_detection_pipeline.py - Implemets the entire pipeline
    
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

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

We can clearly see distortion at the top of the image due to the curved lenses corrected after applying `cv2.undistort()`


Now let's apply distortion correction to 



###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
