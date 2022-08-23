import sys
sys.path.append('.')
from imports import *

path = 'out/filters/'
name = 'weighted_average'
writer = Writer(path)

# Data
kernel2D = np.array([ 
    [   1,   2,   4,   8,  16,  32,  16,   8,   4,   2,   1], 
    [   2,   4,   8,  16,  32,  64,  32,  16,   8,   4,   2], 
    [   4,   8,  16,  32,  64, 128,  64,  32,  16,   8,   4], 
    [   8,  16,  32,  64, 128, 256, 128,  64,  32,  16,   8], 
    [  16,  32,  64, 128, 256, 512, 256, 128,  64,  32,  16],
    [  32,  64, 128, 256, 512,1024, 512, 256, 128,  64,  32],
    [  16,  32,  64, 128, 256, 512, 256, 128,  64,  32,  16],
    [   8,  16,  32,  64, 128, 256, 128,  64,  32,  16,   8], 
    [   4,   8,  16,  32,  64, 128,  64,  32,  16,   8,   4], 
    [   2,   4,   8,  16,  32,  64,  32,  16,   8,   4,   2], 
    [   1,   2,   4,   8,  16,  32,  16,   8,   4,   2,   1]
                  ], dtype=np.float32)
kernel2D = kernel2D / kernel2D.sum()

kernel1D = np.array([ 1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1], dtype=np.float32)
kernel1D = kernel1D / kernel1D.sum()

# Image example (mono)
source_img= cv2.imread('img/baboon.png', cv2.IMREAD_GRAYSCALE)

filtered_img = cv2.filter2D(source_img, -1, kernel2D)

display = cv2.hconcat([source_img, filtered_img])
if common.args().save:
    writer.save_image(name+'_img.png', display)
else:
    common.showImage(display)


# Image example (color)
source_img = cv2.imread('img/baboon.png')

filtered_img = cv2.filter2D(source_img, -1, kernel2D)

display = cv2.hconcat([source_img, filtered_img])
if common.args().save:
    writer.save_image(name+'_color_img.png', display)
else:
    common.showImage(display)


# 1D example
source_arr = Sample().array()
noise_arr = Sample().addNoise().array()
outlier_arr = Sample().addOutliers().array()

filtered_arr = cv2.filter2D(source_arr, -1, kernel1D)
filtered_noise_arr = cv2.filter2D(noise_arr, -1, kernel1D)
filtered_outlier_arr = cv2.filter2D(outlier_arr, -1, kernel1D)

fig = common.plot_three_samples('Low freq', [source_arr, filtered_arr], 'Low freq + hi freq', [noise_arr, filtered_noise_arr], 'Low freq + outliers', [outlier_arr, filtered_outlier_arr])
if common.args().save:
    writer.save_plot(name+'_graph.png', fig)
else:
    plt.show()
