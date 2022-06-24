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


size = 9
start = (105,100)
end = (start[0] + size, start[1] + size)
pix_size = 50

filter_size = 3
filter_half = np.int32((filter_size-1)/2+1)

original = cv2.imread('img/baboon.png', cv2.IMREAD_GRAYSCALE)
draw = cv2.merge([original,original,original])
cv2.rectangle(draw, start, end, (0,0,255))
cv2.imshow('Example', draw)
cv2.waitKey(0)
cv2.destroyAllWindows()
crop_gray = fn.crop(original, start[0], start[1], size, size)
filler = np.zeros([size*pix_size, pix_size, 3], dtype=np.uint8)

# Convolution demo - average with square element
filtered = crop_gray.copy()
for j in range(0, crop_gray.shape[0]-filter_size+1):
    for i in range(0, crop_gray.shape[1]-filter_size+1):
        center_i = i+filter_half-1 
        center_j = j+filter_half-1
        values, count, result = fn.convolution_step(crop_gray, filtered, center_i, center_j, filter_size, filter_size)
        print(str(values) + ' / ' + str(count) + ' = ' + str(result))
        highlight_src = fn.highlight_pixels(crop_gray, i, j, filter_size, filter_size)
        highlight_dst = fn.highlight_pixels(filtered, center_i, center_j, 1, 1)
        zoom_src = fn.zoom(highlight_src, pix_size) 
        zoom_dst = fn.zoom(highlight_dst, pix_size) 
        display_src = fn.annotate_filter(zoom_src, i, j, filter_size, filter_size, pix_size)
        #display_dst = fn.annotate_filter(zoom_dst, center_i, center_j, 1, 1, pix_size)
        display_dst = fn.annotate_rgb(zoom_dst, pix_size)   
        display = cv2.hconcat([display_src, filler, display_dst])
        cv2.imshow('Example1', display)
        cv2.waitKey(100)
cv2.waitKey(0)

# Convolution demo - two-pix_size average with 1D elements = square element
filtered1 = crop_gray.copy()
filtered2 = crop_gray.copy()
for j in range(0, crop_gray.shape[0]):
    for i in range(0, crop_gray.shape[1]-filter_size+1):
        center_i = i+filter_half-1
        values, count, result = fn.convolution_step(crop_gray, filtered1, center_i, j, filter_size, 1)
        print('0: ' + str(values) + ' / ' + str(count) + ' = ' + str(result))
        highlight_src = fn.highlight_pixels(crop_gray, i, j, filter_size, 1)
        highlight_mid = fn.highlight_pixels(filtered1, center_i, j, 1, 1)                
        zoom_src = fn.zoom(highlight_src, pix_size) 
        zoom_mid = fn.zoom(highlight_mid, pix_size) 
        display_src = fn.annotate_filter(zoom_src, i, j, filter_size, 1, pix_size)
        display_mid = fn.annotate_filter(zoom_mid, center_i, j, 1, 1, pix_size)
        display_dst = np.zeros(display_mid.shape, dtype=np.uint8)

        display = cv2.hconcat([display_src, filler, display_mid, filler, display_dst])
        cv2.imshow('Example2', display)
        cv2.waitKey(100)
cv2.waitKey(0)
for j in range(0, crop_gray.shape[0]-filter_size+1):
    for i in range(0, crop_gray.shape[1]):
        center_j = j+filter_half-1
        values, count, result = fn.convolution_step(filtered1, filtered2, i, center_j, 1, filter_size)
        print('1: ' + str(values) + ' / ' + str(count) + ' = ' + str(result))
        highlight_mid = fn.highlight_pixels(filtered1, i, j, 1, filter_size)  
        highlight_dst = fn.highlight_pixels(filtered2, i, center_j, 1, 1)
        zoom_mid = fn.zoom(highlight_mid, pix_size) 
        zoom_dst = fn.zoom(highlight_dst, pix_size) 
        display_mid = fn.annotate_filter(zoom_mid, i, j, 1, filter_size, pix_size)
        #display_dst = fn.annotate_filter(zoom_dst, i, center_j, 1, 1, pix_size) 
        display_dst = fn.annotate_rgb(zoom_dst, pix_size)   
        
        display = cv2.hconcat([display_src, filler, display_mid, filler, display_dst])
        cv2.imshow('Example2', display)
        cv2.waitKey(100)
cv2.waitKey(0)
cv2.destroyAllWindows()