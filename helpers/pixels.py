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
from cvdemo.helpers import functions as fn

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', action='store_true')
args = parser.parse_args()


size = 20
start = (620,205)
end = (start[0] + size, start[1] + size)
step = 50

# Grayscale
original = cv2.imread('img/tulips.png', cv2.IMREAD_GRAYSCALE)
crop = fn.crop(original, start[0], start[1], size, size)
zoom = fn.zoom(crop, step) 
rgb = fn.annotate_gray(zoom, step)

draw = cv2.merge([original,original,original])
draw = cv2.rectangle(draw, start, end, (0,0,200), 1)

crop_rgb = cv2.merge([zoom,zoom,zoom])
display = cv2.hconcat([crop_rgb, rgb])
if args.save:
    save_image('out/other/', 'grayscale_pixels_img.png', display)
    save_image('out/other/', 'grayscale_pixels_rect_img.png', draw)
else:
    cv2.imshow('Example', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# RGB
original = cv2.imread('img/tulips.png')
crop = fn.crop(original, start[0], start[1], size, size)
zoom = fn.zoom(crop, step) 

rgb_r, rgb_g, rgb_b = fn.annotate_channels(zoom, step)

draw = cv2.rectangle(original, start, end, (255,255,255))

grid = cv2.hconcat([cv2.vconcat([zoom, rgb_r]),cv2.vconcat([rgb_g, rgb_b])])

if args.save:
    save_image('out/other/', 'color_pixels_img.png', grid)
    save_image('out/other/', 'color_pixels_rect_img.png', draw)
else:
    cv2.imshow('Example', grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
