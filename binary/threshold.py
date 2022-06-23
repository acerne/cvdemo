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
original = cv2.imread('img/sudoku.png', cv2.IMREAD_GRAYSCALE)
ret, binary = cv2.threshold(original, 63, 255, cv2.THRESH_BINARY)


display = cv2.hconcat([original, binary])
if args.save:
    save_image('out/binary/', 'threshold_img.png', display)
else:
    cv2.imshow('Example', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 1D example
array = polynom_sample()
threshold = np.ones(array.size)*127
ret, binary = cv2.threshold(array, 127, 255, cv2.THRESH_BINARY)


fig = plot_one('Threshold', [array, binary, threshold])
if args.save:
    save_plot('out/binary/', 'threshold_graph.png', fig)
else:
    plt.show()
