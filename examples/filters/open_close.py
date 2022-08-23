import sys
sys.path.append('.')
from imports import *

path = 'out/filters/'
name = 'open_close'
writer = Writer(path)

# Image example (mono)
source_img= cv2.imread('img/baboon.png', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((3, 3), np.uint8)
opened_img = cv2.morphologyEx(source_img, cv2.MORPH_OPEN, kernel)
closed__img = cv2.morphologyEx(source_img, cv2.MORPH_CLOSE, kernel)

display = cv2.hconcat([opened_img, source_img, closed__img])
if common.args().save:
    writer.save_image(name+'_img.png', display)
else:
    common.showImage(display)


# Image example (color)
source_img = cv2.imread('img/baboon.png')

kernel = np.ones((3, 3), np.uint8)
opened_img = cv2.morphologyEx(source_img, cv2.MORPH_OPEN, kernel)
closed__img = cv2.morphologyEx(source_img, cv2.MORPH_CLOSE, kernel)

display = cv2.hconcat([opened_img, source_img, closed__img])
if common.args().save:
    writer.save_image(name+'_color_img.png', display)
else:
    common.showImage(display)


# 1D example
source_arr = Sample().array()
noise_arr = Sample().addNoise().array()
outlier_arr = Sample().addOutliers().array()

kernel = np.ones((11, 1), np.uint8)
opened_arr = cv2.morphologyEx(source_arr, cv2.MORPH_OPEN, kernel)
closed__arr = cv2.morphologyEx(source_arr, cv2.MORPH_CLOSE, kernel)
opened_noise__arr = cv2.morphologyEx(noise_arr, cv2.MORPH_OPEN, kernel)
closed__noise__arr = cv2.morphologyEx(noise_arr, cv2.MORPH_CLOSE, kernel)
opened_outlier_arr = cv2.morphologyEx(outlier_arr, cv2.MORPH_OPEN, kernel)
closed__outlier_arr = cv2.morphologyEx(outlier_arr, cv2.MORPH_CLOSE, kernel)

fig = common.plot_three_samples('Low freq', [source_arr, opened_arr], 'Low freq + hi freq', [noise_arr, opened_noise__arr], 'Low freq + outliers', [outlier_arr, opened_outlier_arr])
if common.args().save:
    writer.save_plot('opening_graph.png', fig)
else:
    plt.show()

fig = common.plot_three_samples('Low freq', [source_arr, closed__arr], 'Low freq + hi freq', [noise_arr, closed__noise__arr], 'Low freq + outliers', [outlier_arr, closed__outlier_arr])
if common.args().save:
    writer.save_plot('closing_graph.png', fig)
else:
    plt.show()


