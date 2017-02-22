# coding=utf-8
import os
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip
from IPython.display import HTML
from find_lane_lines import draw_lane_lines
from find_lane_lines import reset_cache
from find_lane_lines import draw_lane_lines_with_cache

test_mvs_path = os.listdir('raw_movies/')
for movie_path in test_mvs_path:
    print("Movie: " + movie_path)

    # Generate a temporary directory
    tmp_dir = 'lane_line_movies'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Load movie file.
    clip = VideoFileClip('raw_movies/' + movie_path)

    # Generate new lane lines clip.
    reset_cache()
    new_clip1 = clip.fl_image(draw_lane_lines_with_cache)
    new_clip2 = clip.fl_image(draw_lane_lines)

    # Save new video file.
    new_clip1.write_videofile('new_cached_' + movie_path)
    new_clip2.write_videofile('new_nocached_' + movie_path)

