import sys
sys.path.append('.')
from imports import *

path = 'out/level/'
name = 'nonlinear'
writer = Writer(path)

# Data
def sigmoid_curve(X):
    Y = []
    for x in X:
        Y.append(255.0 / (1.0 + np.exp(-(x-127)/16)))
    return Y

lut_identity = np.arange(256, dtype = float)
lut_sigmoid = sigmoid_curve(lut_identity)


# Image example (mono)
source_img= cv2.imread('img/baboon.png', cv2.IMREAD_GRAYSCALE)

transformed_img = cv2.LUT(source_img, np.uint8(lut_sigmoid))

display = cv2.hconcat([source_img, transformed_img])
if common.args().save:
    writer.save_image(name+'_img.png', display)
else:
    common.showImage(display)


# Image example (color)
source_img = cv2.imread('img/baboon.png')

transformed_img = cv2.LUT(source_img, np.uint8(lut_sigmoid))

display = cv2.hconcat([source_img, transformed_img])
if common.args().save:
    writer.save_image(name+'_color_img.png', display)
else:
    common.showImage(display)


# 1D example
source_arr = Sample().array()

transformed_arr = cv2.LUT(source_arr, np.uint8(lut_sigmoid))

fig = common.plot_sample('Nonlinear transform', [[source_arr, 'sample'], [transformed_arr, 'transformed']])
if common.args().save:
    writer.save_plot(name+'_graph.png', fig)
else:
    plt.show()


# Lookup table
fig = common.plot_lut([[lut_identity, 'identity'], [lut_sigmoid, 'sigmoid curve']])
if common.args().save:
    writer.save_plot(name+'_lut.png', fig)
else:
    plt.show()
