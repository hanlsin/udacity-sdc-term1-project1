# coding=utf-8
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Default values of Hough Line Transform.
DFLT_HLPT_RHO = 2
DFLT_HLPT_THETA = np.pi / 180
DFLT_HLPT_THRESHOLD = 10
DFLT_HLPT_MIN_LINE_LENGTH = 20
DFLT_HLPT_MAX_LINE_GAP = 10


def read_image(image_path):
    # NOTE: When using 'mpimg.imread', an error occurs at 'cv2.Canny'.
    #       For this reason, changed to using 'cv2.imread'
    # return mpimg.imread
    return cv2.imread(image_path)


def convert_to_gray(image):
    """
    Convert image to the gray scale image. When you display the converted
    image, you have to set "gray" to show as gray:
        ex. matplotlib.pyplot.imshow(converted_image, cmap="gray"
    :param image: before image
    :return: converted image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def run_gaussian_smoothing(image, kernel_size=5):
    """
    Run Gaussian smoothing of OpenCV and return blurred image.
    http://docs.opencv.org/3.2.0/d4/d13/tutorial_py_filtering.html

    :param image: before image
    :param kernel_size: The width and height of kernel. Should be positive and
    odd.
    :return: blurred image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def run_canny_edge_detection(image, min_threshold=100, max_threshold=150):
    """
    Run Canny edge detection of OpenCV and return the image presenting edges.
    http://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
    :param image: before image
    :param min_threshold: The edge is above this, considered as valid edge.
    :param max_threshold: The edge is above this, considered as sure edge.
    :return: converted edges image
    """
    return cv2.Canny(image, min_threshold, max_threshold)


def mask_rectangle_img(image, left_bottom, left_top, right_top, right_bottom,
                       ignore_mask_color=255):
    """
    Mask image with zero(null) except for the rectangle space. And show
    only ignore_mask_color in the rectangle space.
    :param image: before image
    :param left_bottom: left bottom point of rectangle
    :param left_top: left top point of rectangle
    :param right_top: right bottom point of rectangle
    :param right_bottom: right top point of rectangle
    :param ignore_mask_color: only show this color in the rectangle space
    :return: masked image
    """
    mask_img = np.zeros_like(image)
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]],
                        dtype=np.int32)
    cv2.fillPoly(mask_img, vertices, ignore_mask_color)
    return cv2.bitwise_and(image, mask_img)


def run_hough_lp_transform(image, rho=DFLT_HLPT_RHO, theta=DFLT_HLPT_THETA,
                           threshold=DFLT_HLPT_THRESHOLD,
                           min_line_length=DFLT_HLPT_MIN_LINE_LENGTH,
                           max_line_gap=DFLT_HLPT_MAX_LINE_GAP):
    return cv2.HoughLinesP(image, rho, theta, threshold,
                           min_line_length, max_line_gap)


def draw_line_segments(image, lines_image, line_color=(0, 0, 255),
                       line_thick=3, rho=DFLT_HLPT_RHO,
                       theta=DFLT_HLPT_THETA, threshold=DFLT_HLPT_THRESHOLD,
                       min_line_length=DFLT_HLPT_MIN_LINE_LENGTH,
                       max_line_gap=DFLT_HLPT_MAX_LINE_GAP):
    lines = run_hough_lp_transform(image, rho, theta, threshold,
                                   min_line_length, max_line_gap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lines_image, (x1, y1), (x2, y2), line_color, line_thick)
            cv2.line(lines_image, (x1, y1), (x2, y2), line_color, line_thick)
    return lines_image


def draw_regression_line(image, lines_image, line_color=(255, 0, 0),
                         line_thick=3, rho=DFLT_HLPT_RHO,
                         theta=DFLT_HLPT_THETA, threshold=DFLT_HLPT_THRESHOLD,
                         min_line_length=DFLT_HLPT_MIN_LINE_LENGTH,
                         max_line_gap=DFLT_HLPT_MAX_LINE_GAP):
    lines = run_hough_lp_transform(image, rho, theta, threshold,
                                   min_line_length, max_line_gap)
    try:
        l_fit, r_fit = find_regression_line(lines)
    except ValueError as e:
        raise ValueError(e)

    l_fn = np.poly1d(l_fit)
    r_fn = np.poly1d(r_fit)

    inter_x = find_intersection_x(l_fit, r_fit)
    inter_y = l_fn(inter_x)

    # Draw left regression line.
    cv2.line(lines_image, (0, int(l_fn(0))), (int(inter_x), int(inter_y)),
             line_color, line_thick)
    # Draw right regression line.
    cv2.line(lines_image, (int(inter_x), int(inter_y)),
             (int(image.shape[1]), int(r_fn(image.shape[1]))),
             line_color, line_thick)
    return lines_image


def combine_images(image1, image2, image1_weight=0.8, image2_weight=1):
    return cv2.addWeighted(image1, image1_weight, image2, image2_weight, 0)


def find_regression_line(lines):
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue

            slope = float((y2 - y1)) / float((x2 - x1))
            if slope < -0.2:
                left_line_x.append(x1)
                left_line_y.append(y1)
                left_line_x.append(x2)
                left_line_y.append(y2)
            elif slope > 0.2:
                right_line_x.append(x1)
                right_line_y.append(y1)
                right_line_x.append(x2)
                right_line_y.append(y2)
                # else:
                #    print("ignore line: ")
                #    print([x1, y1, x2, y2])

    if len(left_line_x) == 0 or len(right_line_x) == 0:
        # print("LEFT:")
        # print(left_line_x)
        # print("RIGHT:")
        # print(right_line_x)
        raise ValueError('empty fit')

    return np.polyfit(left_line_x, left_line_y, 1), \
           np.polyfit(right_line_x, right_line_y, 1)


def find_intersection_x(left_fit, right_fit):
    l_m = left_fit[0]
    l_b = left_fit[1]
    r_m = right_fit[0]
    r_b = right_fit[1]
    return (r_b - l_b) / (l_m - r_m)


def draw_lane_lines(image):
    # Convert to gray.
    gray_img = convert_to_gray(image)

    # Run Gaussian smoothing.
    blur_img = run_gaussian_smoothing(gray_img)

    # Run Canny edge detection.
    edges_img = run_canny_edge_detection(blur_img)

    # Mask image except for rectangle space.
    img_shape = image.shape
    masked_img = mask_rectangle_img(edges_img, (50, img_shape[0]), (570, 300),
                                    (620, 300), (img_shape[1], img_shape[0]))
    # masked_img = mask_rectangle_img(edges_img, (50, img_shape[0]), (450, 330),
    #                                (520, 330), (img_shape[1], img_shape[0]))

    # Run Hough line transform and draw regression line.
    try:
        reg_line_img = draw_regression_line(masked_img, np.copy(image) * 0,
                                            line_thick=10)
    except ValueError as e:
        # plt.imshow(gray_img)
        # plt.show()
        # plt.imshow(blur_img)
        # plt.show()
        # plt.imshow(edges_img)
        # plt.show()
        # plt.imshow(masked_img)
        # plt.show()
        return image

    # Combine original image with line segments image.
    return combine_images(image, reg_line_img)


cached_lines_max_size = 100
cached_lines_list = []
cached_lines_idx = 0


def set_cache_size(size):
    cached_lines_max_size = size


def reset_cache():
    cached_lines_list[:] = []
    cached_lines_idx = 0


def draw_lane_lines_with_cache(image):
    # Convert to gray.
    gray_img = convert_to_gray(image)

    # Run Gaussian smoothing.
    blur_img = run_gaussian_smoothing(gray_img)

    # Run Canny edge detection.
    edges_img = run_canny_edge_detection(blur_img)

    # Mask image except for rectangle space.
    img_shape = image.shape
    masked_img = mask_rectangle_img(edges_img, (50, img_shape[0]), (570, 300),
                                    (620, 300), (img_shape[1], img_shape[0]))
    # masked_img = mask_rectangle_img(edges_img, (50, img_shape[0]), (450, 330),
    #                                (520, 330), (img_shape[1], img_shape[0]))

    # Run Hough line transform.
    lines = run_hough_lp_transform(masked_img)

    # Append previous lines.
    global cached_lines_list
    global cached_lines_idx

    if len(cached_lines_list) < cached_lines_max_size:
        cached_lines_list.append(lines[0])
    else:
        cached_lines_list[cached_lines_idx] = lines[0]
    cached_lines_idx = cached_lines_idx + 1
    if cached_lines_idx == cached_lines_max_size:
        cached_lines_idx = 0

    # Draw regression line.
    try:
        l_fit, r_fit = find_regression_line(cached_lines_list)
    except ValueError as e:
        cached_lines_idx = cached_lines_idx - 1

    l_fn = np.poly1d(l_fit)
    r_fn = np.poly1d(r_fit)

    inter_x = find_intersection_x(l_fit, r_fit)
    inter_y = l_fn(inter_x)

    # Draw left regression line.
    reg_line_img = np.copy(image) * 0
    cv2.line(reg_line_img, (0, int(l_fn(0))), (int(inter_x), int(inter_y)),
             (255, 0, 0), 10)
    # Draw right regression line.
    cv2.line(reg_line_img, (int(inter_x), int(inter_y)),
             (int(image.shape[1]), int(r_fn(image.shape[1]))),
             (255, 0, 0), 10)

    # Combine original image with line segments image.
    return combine_images(image, reg_line_img)
