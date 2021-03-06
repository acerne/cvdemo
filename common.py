import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os


def bgr2rgb(img):
    b, g, r = cv2.split(img)
    return cv2.merge([r, g, b])


def polynom_sample():
    # Returns some made up polinomial representation of a 1D sample
    seed = [0, 5, 3, 1, 7, 7, 6, 9, 4, 3, 9, 6, 6, 4, 2, 5]
    seed_y = np.multiply(seed, 255/10)
    seed_x = np.linspace(0, 100, seed_y.size)

    polynom = np.poly1d(np.polyfit(seed_x, seed_y, 8))

    range = np.linspace(0, 100, 100)

    return polynom(range).astype(np.uint8)

def save_image(path, filename, image):
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(path+filename, image)

def plot_one(title, arrays):
    plt.rcParams['figure.figsize'] = [4,4]
    fig, axis = plt.subplots()
    for array in arrays:
        axis.plot(array, linewidth=2, linestyle='-')
    axis.set_title(title)
    return fig

def plot_two(title1, arrays1, title2, arrays2):
    plt.rcParams['figure.figsize'] = [8,4]
    fig, axis = plt.subplots(nrows=1, ncols=2)
    for array in arrays1:
        axis[0].plot(array, linewidth=2, linestyle='-')
    axis[0].set_title(title1)
    for array in arrays2:
        axis[1].plot(array, linewidth=2, linestyle='-')
    axis[1].set_title(title2)
    return fig

def save_plot(path, filename, fig):
    if not os.path.exists('out/binary/'):
        os.makedirs('out/binary/')
    fig.savefig(path+filename, dpi=200)
