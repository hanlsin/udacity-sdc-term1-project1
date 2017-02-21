# coding=utf-8
import os
import matplotlib.pyplot as plt
import numpy as np

from find_lane_lines import draw_lane_lines
from find_lane_lines import read_image
from find_lane_lines import convert_to_gray
from find_lane_lines import run_gaussian_smoothing
from find_lane_lines import run_canny_edge_detection
from find_lane_lines import mask_rectangle_img
from find_lane_lines import draw_line_segments
from find_lane_lines import draw_regression_line
from find_lane_lines import combine_images

test_img_path = os.listdir('raw_images/')
#test_img_path = ['challenge_00000.png']
for img_path in test_img_path:
    print("Image: " + img_path)

    # Read image
    org_img = read_image('raw_images/' + img_path)
    plt.imshow(org_img)
    #plt.show()

    # Convert to gray.
    gray_img = convert_to_gray(org_img)
    plt.imshow(gray_img, cmap="gray")
    #plt.show()

    # Run Gaussian smoothing.
    blur_img = run_gaussian_smoothing(gray_img)
    plt.imshow(blur_img, cmap="gray")
    #plt.show()

    # Run Canny edge detection.
    edges_img = run_canny_edge_detection(blur_img)
    plt.imshow(edges_img)
    #plt.show()

    # Mask image except for rectangle space.
    img_shape = org_img.shape
    masked_img = mask_rectangle_img(edges_img, (50, img_shape[0]), (570, 350),
                                    (620, 350), (img_shape[1], img_shape[0]))
    plt.imshow(masked_img)
    plt.show()

    # Run Hough line transform and draw line segments.
    line_segs_img = draw_line_segments(masked_img, np.copy(org_img) * 0)
    plt.imshow(line_segs_img)
    #plt.show()

    # Combine original image with line segments image.
    comb_lines_img = combine_images(org_img, line_segs_img)
    plt.imshow(comb_lines_img)
    #plt.show()

    # Run Hough line transform and draw regression line.
    try:
        reg_line_img = draw_regression_line(masked_img, np.copy(org_img) * 0,
                                            line_thick=10)
        plt.imshow(reg_line_img)
        #plt.show()
    except Exception as e:
        print e
        continue

    # Combine original image with lane lines image.
    comb_lines_img = combine_images(org_img, reg_line_img)
    plt.imshow(comb_lines_img)
    plt.show()

"""
for img_path in test_img_path:
    print("Image: " + img_path)
    plt.imshow(draw_lane_lines(read_image('raw_images/' + img_path)))
    plt.show()
"""
