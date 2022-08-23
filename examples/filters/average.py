import sys
sys.path.append('.')
from imports import *

path = 'out/filters/'
name = 'average'
writer = Writer(path)

# Image example (mono)
source_img= cv2.imread('img/baboon.png', cv2.IMREAD_GRAYSCALE)

filter_size = (11, 11)
filtered_img = cv2.blur(source_img, filter_size)

display = cv2.hconcat([source_img, filtered_img])
if common.args().save:
    writer.save_image(name+'_img.png', display)
else:
    common.showImage(display)


# Image example (color)
source_img = cv2.imread('img/baboon.png')

filter_size = (11, 11)
filtered_img = cv2.blur(source_img, filter_size)

display = cv2.hconcat([source_img, filtered_img])
if common.args().save:
    writer.save_image(name+'_color_img.png', display)
else:
    common.showImage(display)


# 1D example
source_arr = Sample().array()
noise_arr = Sample().addNoise().array()
outlier_arr = Sample().addOutliers().array()

filter_size = (1, 11)
filtered_arr = cv2.blur(source_arr, filter_size)
filtered_noise_arr = cv2.blur(noise_arr, filter_size)
filtered_outlier_arr = cv2.blur(outlier_arr, filter_size)

fig = common.plot_three_samples('Low freq', [source_arr, filtered_arr], 'Low freq + hi freq', [noise_arr, filtered_noise_arr], 'Low freq + outliers', [outlier_arr, filtered_outlier_arr])
if common.args().save:
    writer.save_plot(name+'_graph.png', fig)
else:
    plt.show()