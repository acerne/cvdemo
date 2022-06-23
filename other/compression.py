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
original = cv2.imread('img/baboon.png')

# JPG compression
# IMWRITE_JPEG_QUALITY - compression percentage: 0 - 100, 0 being the most amount of jpeg compression
filename = 'compressed.jpg'
cv2.imwrite(filename, original, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
jpg = cv2.imread(filename)
os.remove(filename)

diff_16 = cv2.subtract(np.int16(original), np.int16(jpg))
diff_16 = np.absolute(diff_16)
#diff_16 = diff_16 + 127
diff = np.uint8(diff_16)

display = cv2.hconcat([original, diff, jpg])
if args.save:
    save_image('out/other/', 'jpg_diff_img.png', display)
else:
    cv2.imshow('Example', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# PNG compression
# IMWRITE_PNG_COMPRESSION - compression level: 1 - 9, 9 being the most amount of png compression
filename = 'compressed.png'
cv2.imwrite(filename, original, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
png = cv2.imread(filename)
os.remove(filename)

diff_16 = cv2.subtract(np.int16(original), np.int16(png))
diff_16 = np.absolute(diff_16)
diff_16 = diff_16 + 127
diff = np.uint8(diff_16)


display = cv2.hconcat([original, diff, png])
if args.save:
    save_image('out/other/', 'png_diff_img.png', display)
else:
    cv2.imshow('Example', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
