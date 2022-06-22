import cv2
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import numpy as np

import argparse
import sys
sys.path.append('../')
from cvdemo.common import *

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', action='store_true')
args = parser.parse_args()


# Image example
original = cv2.imread('img/stars.png')

kernel = np.ones((3, 3), np.uint8)
result_open = cv2.morphologyEx(original, cv2.MORPH_OPEN, kernel)
result_close = cv2.morphologyEx(original, cv2.MORPH_CLOSE, kernel)


display = cv2.hconcat([result_open, original, result_close])
if args.save:
    save_image('out/filters/', 'open_close_img.png', display)
else:
    cv2.imshow('Example', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 1D example
array = polynom_sample()

kernel = np.ones((11, 1), np.uint8)
array_open = cv2.morphologyEx(array, cv2.MORPH_OPEN, kernel)
array_close = cv2.morphologyEx(array, cv2.MORPH_CLOSE, kernel)


fig = plot_two(array, array_open, 'Opening', array_close, 'Closing')
if args.save:
    save_plot('out/filters/', 'open_close_graph.png', fig)
else:
    plt.show()

