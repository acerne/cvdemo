import cv2
import matplotlib.pyplot as plt
import argparse

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', action='store_true')
    return parser.parse_args()

def bgr2rgb(img):
    b, g, r = cv2.split(img)
    return cv2.merge([r, g, b])

def showImage(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_sample(title, arrays):
    plt.rcParams['figure.figsize'] = [5,5]
    fig, axis = plt.subplots()
    for array in arrays:
        axis.plot(array[0], linewidth=2, linestyle='-', label=array[1])
    axis.set_title(title)
    axis.legend()
    axis.get_xaxis().set_visible(False)
    axis.set_ylim(-5, 260)
    axis.set_yticks([0,63,127,191,255])
    axis.set_ylabel('Value [uint8]')
    return fig

def plot_two_samples(title1, arrays1, title2, arrays2):
    plt.rcParams['figure.figsize'] = [10,5]
    fig, axis = plt.subplots(nrows=1, ncols=2)
    for array in arrays1:
        axis[0].plot(array, linewidth=2, linestyle='-')
    axis[0].set_title(title1)
    for array in arrays2:
        axis[1].plot(array, linewidth=2, linestyle='-')
    axis[1].set_title(title2)
    for ax in axis:
        ax.get_xaxis().set_visible(False)
        ax.set_ylim(-2, 257)
        ax.set_yticks([0,63,127,191,255])
    return fig

def plot_three_samples(title1, arrays1, title2, arrays2, title3, arrays3):
    plt.rcParams['figure.figsize'] = [15,5]
    fig, axis = plt.subplots(nrows=1, ncols=3)
    for array in arrays1:
        axis[0].plot(array, linewidth=2, linestyle='-')
    axis[0].set_title(title1)
    for array in arrays2:
        axis[1].plot(array, linewidth=2, linestyle='-')
    axis[1].set_title(title2)
    for array in arrays3:
        axis[2].plot(array, linewidth=2, linestyle='-')
    axis[2].set_title(title3)
    for ax in axis:
        ax.get_xaxis().set_visible(False)
        ax.set_ylim(-2, 257)
        ax.set_yticks([0,63,127,191,255])
    return fig

def plot_lut(arrays):
    plt.rcParams['figure.figsize'] = [5,5]
    fig, axis = plt.subplots()
    for array in arrays:
        axis.plot(array[0], linewidth=2, linestyle='-', label=array[1])
    axis.set_title('Lookup table')
    axis.legend()
    axis.set_xlim(-2, 257)
    axis.set_ylim(-2, 257)
    axis.set_xticks([0,63,127,191,255])
    axis.set_yticks([0,63,127,191,255])
    axis.set_xlabel('Input value [uint8]')
    axis.set_ylabel('Output value [uint8]')
    return fig