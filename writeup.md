#**Finding Lane Lines on the Road** 

##Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[image1]: fll_whiteCarLaneSwitch.jpg

---

### Reflection

#### Pipeline Steps
My pipeline function, 'draw_lane_lines', in the source file, 'find_lane_lines.py', consisted of 6 steps below:

1. Convert an original image to gray.
Using OpenCV, a gray scale image is returned from 'convert_to_gray' function.

2. Run Gaussian smoothing.
Using Gaussian smoothing function of OpenCV, a blured gray image is returned from 'run_gaussian_smoothing' function with the gray image from STEP 1. When using this function, the second parameter, 'kernel_size', should be positive and odd.

3. Run Canny edge detection.
Using Canny function of OpenCV, edges that are detected from the blured gray image of STEP 2 are appeared in a new image produced by 'run_canny_edge_detection' function. Default values of minimum threshold and maximum threshold are 100 and 150, and you can change these as parameters.

4. Mask edged image except for a rectangle space.
Detected all edges in the image are not necessary, so make a quadrangle space for interesting edges and fill the rest with black. Default shape of the interesting space is a trapezium.

5. Draw regression lines using line segments from Hough transform.
Using Hough transform of OpenCV, line segments are found from 'run_hough_lp_transform' with the masked edges image. Method for regression lines are explained in detail next. As a default, line segments are showed as red lines. Default parameter values for Hough transform:
 * rho = 2
 * theta = pi / 180
 * threshold = 10
 * minimum line length = 20
 * maximum line gap = 10

6. Combine the original image and the regression lines.
In the original image, red regression lines are showed up from 'combine_images' function.

#### Method of Regression Lines
In 'draw_regression_line' function, the line segments data of Hough transform, which consists of 2 points, x1, y1, x2, and y2, are used to find regression lines.

First of all, the data have to be separated by a left line, right line, or ignore. As I said, the slope of one line segment can be caculated and be categorized as:
* slope < -0.2         : left line
* slope > 0.2          : right line
* -0.2 <= slope <= 0.2 : ignore

Secondly, linear equations for separated left and right are obtained by 'poly1d' function of Numpy. Using two equations, find the intersection point, and use this point as the end of two lines.

In addition, I applied a cache technique for line segments data of Hough transform to increase accuracy of regression lines. The cache is useful only for movies, because images doen't have any data for the cache. A movie is a set of sequential images, so previous images can be used for finding regression lines. The cache saves line segments data of previous images. This technique can prevent appearing suddenly a skewed line. You can find the effect of the cache in 'find_lane_lines.ipynb'.

#### Result Image
The test image file, 'whiteCarLaneSwitch.jpg', is showed step by step as below:
![alt text][image1]


#### Potential Shortcomings
1. If edges in masked image are showed only a line, 'draw_regression_line' function throws an exception, and fail to draw regression lines.
2. The rectangle space is not auto-detected, so if a camera for images or movies is putted in a different location of a car, this pipeline could go wrong.

#### Suggest Possible Improvements
When using the cache, cached values are used as a same weight. So, when a cache size is too big and a car meets a swerved corner, the degree of found lane lines could be too small to go through the corner. However, if each cached value has a weight and the weight influences to find a regression line, it would make an accurate lane line.
