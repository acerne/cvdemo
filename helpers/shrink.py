import cv2

import sys
import os
sys.path.append('../')
from cvdemo.common import *

src = cv2.imread('img/space.jpg')

#percent by which the image is resized
scale_percent = 50

#calculate the 50 percent of original dimensions
width = int(src.shape[1] * scale_percent / 100)
height = int(src.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

# resize image
output = cv2.resize(src, dsize)

cv2.imwrite('img/space.png', output)
