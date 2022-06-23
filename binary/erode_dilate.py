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
original = cv2.imread('img/fingerprint.png', cv2.IMREAD_GRAYSCALE)
ret, binary = cv2.threshold(original, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)
result_erode = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel)
result_dilate = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)


display = cv2.hconcat([result_erode, binary, result_dilate])
if args.save:
    save_image('out/binary/', 'erode_dilate_img.png', display)
else:
    cv2.imshow('Example', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 1D example
array = polynom_sample()
ret, binary = cv2.threshold(array, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((11, 1), np.uint8)
array_erode = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel)
array_dilate = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)


fig = plot_two('Erosion', [binary, array_erode], 'Dilation', [binary, array_dilate])
if args.save:
    save_plot('out/binary/', 'erode_dilate_graph.png', fig)
else:
    plt.show()
