import sys
sys.path.append('.')
from imports import *

path = 'out/level/'
name = 'brightness'
writer = Writer(path)

# Image example (mono)
source_img= cv2.imread('img/baboon.png', cv2.IMREAD_GRAYSCALE)
source_img16 = np.int16(source_img)

brighter_img = np.uint8(np.clip(source_img16 + 64, 0, 255))
darker_img = np.uint8(np.clip(source_img16 - 64, 0, 255))

display = cv2.hconcat([darker_img, source_img, brighter_img])
if common.args().save:
    writer.save_image(name+'_img.png', display)
else:
    common.showImage(display)


# Image example (color)
source_img = cv2.imread('img/baboon.png')
source_img16 = np.int16(source_img)

brighter_img = np.uint8(np.clip(source_img16 + 64, 0, 255))
darker_img = np.uint8(np.clip(source_img16 - 64, 0, 255))

display = cv2.hconcat([darker_img, source_img, brighter_img])
if common.args().save:
    writer.save_image(name+'_color_img.png', display)
else:
    common.showImage(display)


# 1D example
source_arr = Sample().array()
source_arr16 = np.int16(source_arr)

brighter_arr = np.uint8(np.clip(source_arr16 + 64, 0, 255))
darker_arr = np.uint8(np.clip(source_arr16 - 64, 0, 255))

fig = common.plot_two_samples('Darker', [source_arr, darker_arr], 'Brighter', [source_arr, brighter_arr])
if common.args().save:
    writer.save_plot(name+'_graph.png', fig)
else:
    plt.show()


# Lookup table
lut_identity = np.arange(256, dtype = float)
lut_bright = np.clip(lut_identity + 64, 0, 255)
lut_dark = np.clip(lut_identity - 64, 0, 255)

fig = common.plot_lut([[lut_identity, 'identity'], [lut_bright, 'brighter'], [lut_dark, 'darker']])
if common.args().save:
    writer.save_plot(name+'_lut.png', fig)
else:
    plt.show()