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
triangle = cv2.imread('img/triangle.png', cv2.IMREAD_GRAYSCALE)
circle = cv2.imread('img/circle.png', cv2.IMREAD_GRAYSCALE)


intersection = cv2.bitwise_and(triangle, circle)

display = cv2.hconcat([triangle, intersection, circle])
if args.save:
    save_image('out/binary/', 'bitwise_intersection_img.png', display)
else:
    cv2.imshow('Example', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


union = cv2.bitwise_or(triangle, circle)

display = cv2.hconcat([triangle, union, circle])
if args.save:
    save_image('out/binary/', 'bitwise_union_img.png', display)
else:
    cv2.imshow('Example', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


difference = cv2.bitwise_xor(triangle, circle)

display = cv2.hconcat([triangle, difference, circle])
if args.save:
    save_image('out/binary/', 'bitwise_difference_img.png', display)
else:
    cv2.imshow('Example', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
