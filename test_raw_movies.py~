# coding=utf-8
import os
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip
from find_lane_lines import draw_lane_lines
from find_lane_lines import reset_cache
from find_lane_lines import draw_lane_lines_with_cache

test_mvs_path = os.listdir('raw_movies/')
for movie_path in test_mvs_path:
    print("Movie: " + movie_path)

    # Generate a temporary directory
    tmp_dir = 'raw_movies/tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Load movie file.
    clip = VideoFileClip('raw_movies/' + movie_path)

    # Generate new lane lines clip.
    reset_cache()
    new_clip = clip.fl_image(draw_lane_lines_with_cache)
    #new_clip = clip.fl_image(draw_lane_lines)

    # Save new video file.
    new_clip.write_videofile('raw_movies/tmp/new_' + movie_path)

