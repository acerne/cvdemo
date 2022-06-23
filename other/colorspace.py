import cv2
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import numpy as np

import argparse
import sys
import os
sys.path.append('../')
from cvdemo.common import *

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', action='store_true')
args = parser.parse_args()


# Image example
original = cv2.imread('img/peppers.png')
b,g,r = cv2.split(original)

z = np.zeros(b.shape, dtype=np.uint8)
b = cv2.merge([b,z,z])
g = cv2.merge([z,g,z])
r = cv2.merge([z,z,r])
display = cv2.hconcat([original, r, g, b])
if args.save:
    save_image('out/other/', 'rgb_img.png', display)
else:
    cv2.imshow('Example', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Colorspace HSV
hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)

h = np.uint8(np.float32(h)/180*255)
h = cv2.applyColorMap(h, cv2.COLORMAP_HSV)
s = cv2.merge([s,s,s])
v = cv2.merge([v,v,v])
display = cv2.hconcat([original, h, s, v])
if args.save:
    save_image('out/other/', 'hsv_img.png', display)
else:
    cv2.imshow('Example', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Back to RGB
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

diff_16 = cv2.subtract(np.int16(original), np.int16(bgr))
diff_16_abs = np.absolute(diff_16)
diff_16_adj = diff_16*10 + 127
diff = np.uint8(diff_16_adj)

print('Sum of absolute differences: ' + str(np.sum(diff_16_abs)))
print('Avg of absolute differences: ' + str(np.average(diff_16_abs)))
print('Max of absolute differences: ' + str(np.max(diff_16_abs)))
display = cv2.hconcat([original, diff, bgr])
if args.save:
    save_image('out/other/', 'rgb_diff_img.png', display)
else:
    cv2.imshow('Example', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()