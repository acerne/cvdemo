import numpy as np

class Sample:
    def __init__(self) -> None:
        # Returns some made up polinomial representation of a 1D sample
        seed = [0, 5, 3, 1, 7, 7, 6, 9, 4, 3, 9, 6, 6, 4, 2, 5]
        seed_y = np.multiply(seed, 255/10)
        seed_x = np.linspace(0, 100, seed_y.size)

        polynom = np.poly1d(np.polyfit(seed_x, seed_y, 8))

        range = np.linspace(0, 100, 100)
        self.data = polynom(range)
    
    def addNoise(self):        
        hi_freq = 10*np.sin(20*np.linspace(-np.pi,np.pi,100))
        self.data += hi_freq
        return self

    def addOutliers(self):
        self.data[20] = 0 # outlier
        self.data[30] = 255 # outlier
        self.data[60] = 0 # outlier
        return self
    
    def array(self):
        return self.data.astype(np.uint8)