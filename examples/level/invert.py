import sys
sys.path.append('.')
from imports import *

path = 'out/level/'
name = 'inverted'
writer = Writer(path)


# Image example (mono)
source_img= cv2.imread('img/baboon.png', cv2.IMREAD_GRAYSCALE)

inverted_img = 255 - source_img

display = cv2.hconcat([source_img, inverted_img])
if common.args().save:
    writer.save_image(name+'_img.png', display)
else:
    common.showImage(display)


# Image example (color)
source_img = cv2.imread('img/baboon.png')
source_img16 = np.int16(source_img)

inverted_img = 255 - source_img

display = cv2.hconcat([source_img, inverted_img])
if common.args().save:
    writer.save_image(name+'_color_img.png', display)
else:
    common.showImage(display)


# 1D example
source_arr = Sample().array()

inverted_arr = 255 - source_arr

fig = common.plot_sample('Inversion', [[source_arr, 'sample'], [inverted_arr, 'inverted']])
if common.args().save:
    writer.save_plot(name+'_graph.png', fig)
else:
    plt.show()


# Lookup table
lut_identity = np.arange(256, dtype = float)
lut_inverted = 255 - lut_identity

fig = common.plot_lut([[lut_identity, 'identity'], [lut_inverted, 'inverted']])
if common.args().save:
    writer.save_plot(name+'_lut.png', fig)
else:
    plt.show()
