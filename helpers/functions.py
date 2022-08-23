import cv2
import numpy as np

def crop(img, x, y, w, h):
    if len(img.shape) > 2:
        return img[y:(y+h), x:(x+w), :]
    else:
        return img[y:(y+h), x:(x+w)]

def zoom(img, pix_size):
    return cv2.resize(img, [img.shape[1]*pix_size, img.shape[0]*pix_size], interpolation=cv2.INTER_NEAREST)

def annotate_gray(img, pix_size):
    img_copy = img.copy()
    for y in range(0, img.shape[0], pix_size):
        for x in range(0, img.shape[1], pix_size):
            if img[y][x] > 127:
                cv2.putText(img_copy, hex(img[y][x])[2:], (x+5,y+35), cv2.FONT_HERSHEY_COMPLEX, 1, 0)
            else:
                cv2.putText(img_copy, hex(img[y][x])[2:], (x+5,y+35), cv2.FONT_HERSHEY_COMPLEX, 1, 255)
    return img_copy

def annotate_rgb(img, pix_size):
    img_copy = img.copy()
    for y in range(0, img.shape[0], pix_size):
        for x in range(0, img.shape[1], pix_size):
            cv2.putText(img_copy, hex(img[y][x][0])[2:], (x+5,y+35), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
    return img_copy

def annotate_channels(img, pix_size):
    b,g,r = cv2.split(img)
    z = np.zeros(b.shape, dtype=np.uint8)
    rgb_r = cv2.merge([z,z,r])
    rgb_g = cv2.merge([z,g,z])
    rgb_b = cv2.merge([b,z,z])
    for y in range(0, img.shape[0], pix_size):
        for x in range(0, img.shape[1], pix_size):
            cv2.putText(rgb_r, hex(r[y][x])[2:], (x+5,y+35), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
            cv2.putText(rgb_g, hex(g[y][x])[2:], (x+5,y+35), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
            cv2.putText(rgb_b, hex(b[y][x])[2:], (x+5,y+35), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
    return rgb_r, rgb_g, rgb_b

def to_packed(img):
    h = img.shape[0]
    w = img.shape[1]
    w3 = img.shape[1]*3
    packed = np.zeros([h, w3, 3], dtype=np.uint8)
    for y in range(0, h, 1):
        for x in range(0, w, 1):
            packed[y,3*x,2] = img[y,x,2]
            packed[y,3*x+1,1] = img[y,x,1]
            packed[y,3*x+2,0] = img[y,x,0]
    return packed

def annotate_filter(img, i, j, w, h, pix_size):
    img_copy = img.copy()
    for y in range(j*pix_size, j*pix_size+h*pix_size, pix_size):
        for x in range(i*pix_size, i*pix_size+w*pix_size, pix_size):
            cv2.putText(img_copy, hex(img[y][x][0])[2:], (x+5,y+35), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
    return img_copy

def highlight_pixels(img, x, y, w, h):
    img_rgb = cv2.merge([img,img,img])
    for j in range(y, (y+h)):
        for i in range(x, (x+w)):
            img_rgb[j][i][1] = np.clip(img_rgb[j][i][1]+64, 0, 255)
    return img_rgb

def convolution_step(img_src, img_dst, x, y, filter_width, filter_height):
    filter_width_half = np.int32((filter_width-1)/2+1)
    filter_height_half = np.int32((filter_height-1)/2+1)
    values = []
    for j in range(y-filter_height_half+1, y+filter_height_half):
        for i in range(x-filter_width_half+1, x+filter_width_half):
            value = img_src[j][i] 
            values.append(value)
    count = filter_width*filter_height
    result = np.uint8(np.floor(sum(values) / count))
    img_dst[y][x] = result
    return values, count, result
