import sys
sys.path.append('.')
from imports import *

path = 'out/level/'
name = 'contrast'
writer = Writer(path)

# Image example (mono)
source_img= cv2.imread('img/baboon.png', cv2.IMREAD_GRAYSCALE)
source_img16 = np.int16(source_img)

high_contrast_img = np.uint8(np.clip((source_img16 - 127) * 1.5 + 127, 0, 255))
low_contrast_img = np.uint8(np.clip((source_img16 - 127) * 0.5 + 127, 0, 255))

display = cv2.hconcat([low_contrast_img, source_img, high_contrast_img])
if common.args().save:
    writer.save_image(name+'_img.png', display)
else:
    common.showImage(display)


# Image example (color)
source_img = cv2.imread('img/baboon.png')
source_img16 = np.int16(source_img)

high_contrast_img = np.uint8(np.clip((source_img16 - 127) * 1.5 + 127, 0, 255))
low_contrast_img = np.uint8(np.clip((source_img16 - 127) * 0.5 + 127, 0, 255))

display = cv2.hconcat([low_contrast_img, source_img, high_contrast_img])
if common.args().save:
    writer.save_image(name+'_color_img.png', display)
else:
    common.showImage(display)


# 1D example
source_arr = Sample().array()
source_arr16 = np.int16(source_arr)

high_contrast_arr = np.uint8(np.clip((source_arr16 - 127) * 1.5 + 127, 0, 255))
low_contrast_arr = np.uint8(np.clip((source_arr16 - 127) * 0.5 + 127, 0, 255))

fig = common.plot_two_samples('Low contrast', [source_arr, low_contrast_arr], 'High contrast', [source_arr, high_contrast_arr])
if common.args().save:
    writer.save_plot(name+'_graph.png', fig)
else:
    plt.show()


# Lookup table
lut_identity = np.arange(256, dtype = float)
lut_high_contrast = np.clip((lut_identity - 127) * 1.5 + 127, 0, 255)
lut_low_contrast = np.clip((lut_identity - 127) * 0.5 + 127, 0, 255)

fig = common.plot_lut([[lut_identity, 'identity'], [lut_low_contrast, 'low contrast'], [lut_high_contrast, 'high contrast']])
if common.args().save:
    writer.save_plot(name+'_lut.png', fig)
else:
    plt.show()