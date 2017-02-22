# coding=utf-8
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fontm
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

    fig = plt.figure(figsize=(20, 10))
    fig_txt = fig.text(0.5, 0.95, img_path, horizontalalignment='center',
                       fontproperties=fontm.FontProperties(size=15))
    grid_s = (3, 4)
    len_fig = 3

    # Read image
    org_img = read_image('raw_images/' + img_path)
    ax = plt.subplot2grid(grid_s, (0, 0))
    ax.imshow(org_img)
    ax.set_title('Original')

    # Convert to gray.
    gray_img = convert_to_gray(org_img)
    ax = plt.subplot2grid(grid_s, (0, 1))
    ax.imshow(gray_img, cmap="gray")
    ax.set_title('Gray')

    # Run Gaussian smoothing.
    blur_img = run_gaussian_smoothing(gray_img)
    ax = plt.subplot2grid(grid_s, (0, 2))
    ax.imshow(blur_img, cmap="gray")
    ax.set_title('Blur')

    # Run Canny edge detection.
    edges_img = run_canny_edge_detection(blur_img)
    ax = plt.subplot2grid(grid_s, (0, 3))
    ax.imshow(edges_img)
    ax.set_title('Edge')

    # Mask image except for rectangle space.
    img_shape = org_img.shape
    masked_img = mask_rectangle_img(edges_img, (50, img_shape[0]), (470, 350),
                                    (620, 350), (img_shape[1], img_shape[0]))
    ax = plt.subplot2grid(grid_s, (1, 0))
    ax.imshow(masked_img)
    ax.set_title('Masked')

    # Run Hough line transform and draw line segments.
    line_segs_img = draw_line_segments(masked_img, np.copy(org_img) * 0)
    ax = plt.subplot2grid(grid_s, (1, 1))
    ax.imshow(line_segs_img)
    ax.set_title('Segments')

    # Combine original image with line segments image.
    comb_lines_img = combine_images(org_img, line_segs_img)
    ax = plt.subplot2grid(grid_s, (1, 2))
    ax.imshow(comb_lines_img)
    ax.set_title('Combine 1')

    # Run Hough line transform and draw regression line.
    try:
        reg_line_img = draw_regression_line(masked_img, np.copy(org_img) * 0,
                                            line_thick=10)
        ax = plt.subplot2grid(grid_s, (1, 3))
        ax.imshow(reg_line_img)
        ax.set_title('Regression')
    except Exception as e:
        ax = plt.subplot2grid(grid_s, (1, 3))
        ax.imshow(reg_line_img)
        ax.set_title(str(e))
        plt.show()
        continue

    # Combine original image with lane lines image.
    comb_lines_img = combine_images(org_img, reg_line_img)
    ax = plt.subplot2grid(grid_s, (2, 0))
    ax.imshow(comb_lines_img)
    ax.set_title('Combine 2')

    #plt.show()
    plt.savefig('fll_' + img_path)

"""
for img_path in test_img_path:
    print("Image: " + img_path)
    plt.imshow(draw_lane_lines(read_image('raw_images/' + img_path)))
    plt.show()
"""
