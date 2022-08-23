import cv2
import matplotlib
import matplotlib.pyplot as plt
import os

class Writer:
    def __init__(self, path) -> None:
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save_image(self, filename, image):        
        cv2.imwrite(self.path + filename, image)

    def save_plot(self, filename, fig):
        fig.savefig(self.path + filename, dpi=200)
