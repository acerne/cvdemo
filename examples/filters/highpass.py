import sys
sys.path.append('.')
from imports import *

path = 'out/filters/'
name = 'highpass'
writer = Writer(path)

# Data
kernel2D = -np.ones([11, 11], dtype=np.int16)
sum = -kernel2D.sum()
kernel2D[5,5] = sum-1
kernel2D = kernel2D / sum

kernel1D = -np.ones([11], dtype=np.int16)
sum = -kernel1D.sum()
kernel1D[5] = sum-1
kernel1D = kernel1D / sum

# Image example (mono)
source_img= cv2.imread('img/baboon.png', cv2.IMREAD_GRAYSCALE)

filtered_img16 = cv2.filter2D(source_img, cv2.CV_16S, kernel2D)
filtered_img = np.uint8(filtered_img16 + 127)

display = cv2.hconcat([source_img, filtered_img])
if common.args().save:
    writer.save_image(name+'_img.png', display)
else:
    common.showImage(display)


# Image example (color)
source_img = cv2.imread('img/baboon.png')

filtered_img16 = cv2.filter2D(source_img, cv2.CV_16S, kernel2D)
filtered_img = np.uint8(filtered_img16 + 127)

display = cv2.hconcat([source_img, filtered_img])
if common.args().save:
    writer.save_image(name+'_color_img.png', display)
else:
    common.showImage(display)


# 1D example
source_arr = Sample().array()
noise_arr = Sample().addNoise().array()
outlier_arr = Sample().addOutliers().array()

filtered_arr = np.uint8(cv2.filter2D(source_arr, cv2.CV_16S, kernel1D)  + 127)
filtered_noise_arr = np.uint8(cv2.filter2D(noise_arr, cv2.CV_16S, kernel1D)  + 127)
filtered_outlier_arr = np.uint8(cv2.filter2D(outlier_arr, cv2.CV_16S, kernel1D)  + 127)

fig = common.plot_three_samples('Low freq', [source_arr, filtered_arr], 'Low freq + hi freq', [noise_arr, filtered_noise_arr], 'Low freq + outliers', [outlier_arr, filtered_outlier_arr])
if common.args().save:
    writer.save_plot(name+'_graph.png', fig)
else:
    plt.show()
