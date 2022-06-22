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
original = cv2.imread('img/baboon.png')

average = cv2.blur(original, [11, 11])


display = cv2.hconcat([original, average])
if args.save:
    save_image('out/filters/', 'average_img.png', display)
else:
    cv2.imshow('Example', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 1D example
array = polynom_sample()

array_blur = cv2.blur(array, [1, 11])


fig = plot_one(array, array_blur, 'Average')
if args.save:
    save_plot('out/filters/', 'average_graph.png', fig)
else:
    plt.show()