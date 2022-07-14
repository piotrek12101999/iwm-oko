import numpy as np
import cv2
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_minimum, threshold_local
from skimage import img_as_ubyte
from skimage.morphology import dilation, erosion


class BasicDetection:
    def __init__(self, image):
        self.__image = io.imread(image)

    def create_bg_mask(self):
        img = np.uint8(rgb2gray(self.__image.copy()) * 255)
        binary_min = img > threshold_minimum(img)
        return img_as_ubyte(binary_min)

    def compute_expert_mask(self):
        mask = self.create_bg_mask()

        self.__image[:, :, 0] = 0
        self.__image[:, :, 2] = 0

        image = np.uint8(rgb2gray(self.__image.copy()) * 255)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(image)
        binary_adaptive = cl1 > threshold_local(cl1, 213, offset=10)
        binary_adaptive = dilation(dilation(dilation(erosion(binary_adaptive))))
        binary_adaptive[mask == 0] = 1

        result = np.zeros(binary_adaptive.shape)

        result[binary_adaptive == True] = 0
        result[binary_adaptive == False] = 1

        return dilation(result)
