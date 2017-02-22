# coding=utf-8
import os
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip
from find_lane_lines import turn_on_cache, turn_off_cache
from find_lane_lines import draw_lane_lines

test_movies_dir = 'raw_movies/'
test_movie_name_list = os.listdir(test_movies_dir)
test_movie_name_list = ['challenge.mp4']
for movie_name in test_movie_name_list:
    print("Movie: " + movie_name)

    # Load movie file.
    clip = VideoFileClip(test_movies_dir + movie_name)

    # Generate new lane lines clip with cache.
    turn_on_cache()
    new_clip1 = clip.fl_image(draw_lane_lines)
    # Save new video file.
    new_movie1_name = 'new_cached_' + movie_name
    new_clip1.write_videofile(new_movie1_name)
    turn_off_cache()

    # Generate new lane lines clip.
    new_clip2 = clip.fl_image(draw_lane_lines)
    # Save new video file.
    new_movie2_name = 'new_no_cached_' + movie_name
    new_clip1.write_videofile(new_movie2_name)

    # Generate a temporary directory
    tmp_dir = 'tmp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    clip.write_images_sequence(tmp_dir + movie_name + '_%05d.jpg')
