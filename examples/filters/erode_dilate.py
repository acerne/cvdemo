import sys
sys.path.append('.')
from imports import *

path = 'out/filters/'
name = 'erode_dilate'
writer = Writer(path)

# Image example (mono)
source_img= cv2.imread('img/baboon.png', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((3, 3), np.uint8)
eroded_img = cv2.morphologyEx(source_img, cv2.MORPH_ERODE, kernel)
dilated_img = cv2.morphologyEx(source_img, cv2.MORPH_DILATE, kernel)

display = cv2.hconcat([eroded_img, source_img, dilated_img])
if common.args().save:
    writer.save_image(name+'_img.png', display)
else:
    common.showImage(display)


# Image example (color)
source_img = cv2.imread('img/baboon.png')

kernel = np.ones((3, 3), np.uint8)
eroded_img = cv2.morphologyEx(source_img, cv2.MORPH_ERODE, kernel)
dilated_img = cv2.morphologyEx(source_img, cv2.MORPH_DILATE, kernel)

display = cv2.hconcat([eroded_img, source_img, dilated_img])
if common.args().save:
    writer.save_image(name+'_color_img.png', display)
else:
    common.showImage(display)


# 1D example
source_arr = Sample().array()
noise_arr = Sample().addNoise().array()
outlier_arr = Sample().addOutliers().array()

kernel = np.ones((11, 1), np.uint8)
eroded_arr = cv2.morphologyEx(source_arr, cv2.MORPH_ERODE, kernel)
dilated_arr = cv2.morphologyEx(source_arr, cv2.MORPH_DILATE, kernel)
eroded_noise__arr = cv2.morphologyEx(noise_arr, cv2.MORPH_ERODE, kernel)
dilated_noise__arr = cv2.morphologyEx(noise_arr, cv2.MORPH_DILATE, kernel)
eroded_outlier_arr = cv2.morphologyEx(outlier_arr, cv2.MORPH_ERODE, kernel)
dilated_outlier_arr = cv2.morphologyEx(outlier_arr, cv2.MORPH_DILATE, kernel)

fig = common.plot_three_samples('Low freq', [source_arr, eroded_arr], 'Low freq + hi freq', [noise_arr, eroded_noise__arr], 'Low freq + outliers', [outlier_arr, eroded_outlier_arr])
if common.args().save:
    writer.save_plot(name+'_graph.png', fig)
else:
    plt.show()

fig = common.plot_three_samples('Low freq', [source_arr, dilated_arr], 'Low freq + hi freq', [noise_arr, dilated_noise__arr], 'Low freq + outliers', [outlier_arr, dilated_outlier_arr])
if common.args().save:
    writer.save_plot(name+'_graph.png', fig)
else:
    plt.show()



