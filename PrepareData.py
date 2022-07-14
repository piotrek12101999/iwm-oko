import numpy as np
import functools
import cv2
import pandas as pd
import operator
from skimage import filters
from sklearn.utils import resample


class PrepareData:
    slice_size = 7

    @staticmethod
    def __calc_contrast(val, lo, hi):
        if val < lo:
            return 0
        if val > hi:
            return 1
        return (val - lo) / (hi - lo)

    def __contrast_image(self, img, low, high):
        res = [[self.__calc_contrast(x, low, high) for x in y] for y in img]
        return np.array(res)

    def __increase_contrast(self, img):
        nonzero_photo = img[np.nonzero(img)]
        percentiles = np.percentile(nonzero_photo, (2, 99))
        return self.__contrast_image(img, percentiles[0], percentiles[1])

    def __detect_edges(self, img):
        img_sobel = filters.sobel(img)
        percentile = np.percentile(img_sobel, (1, 99))
        return self.__contrast_image(img_sobel, percentile[0], percentile[1])

    def __round_to_slice_size(self, dim):
        return self.slice_size * ((dim - self.slice_size) // self.slice_size)

    def __slice_image(self, image, step):
        slices = []

        for r in range(0, self.__round_to_slice_size(image.shape[0]), step):
            for c in range(0, self.__round_to_slice_size(image.shape[1]), step):
                slices.append(image[r:r + self.slice_size, c:c + self.slice_size])

        return slices

    def __to_hu_moments(self, images):
        hu_list = []
        for img in images:
            moments = cv2.moments(np.vectorize(float)(img))
            hu_moments = cv2.HuMoments(moments)
            hu_list.append(self.flatten(hu_moments))

        return pd.DataFrame(hu_list)

    def __preprocess(self, img):
        img = self.normalize(img)
        img = self.__increase_contrast(img)
        img = self.__detect_edges(img)
        return img

    def __take_samples(self, img, step):
        slices = self.__slice_image(img, step)
        slices = [self.flatten(slice) for slice in slices]
        slices = np.append(slices, self.__to_hu_moments(slices), axis=1)
        return pd.DataFrame(slices)

    def __label_samples(self, img, step):
        slices = self.__slice_image(img, step)
        central_pixels = [np.round(slice[self.slice_size // 2][self.slice_size // 2]) for slice in slices]
        return pd.DataFrame(central_pixels)

    @staticmethod
    def normalize(img):
        return img / np.max(img)

    @staticmethod
    def flatten(arr):
        return functools.reduce(operator.iconcat, arr, [])

    def reshape_labels(self, labels, shape):
        return labels.reshape(self.__round_to_slice_size(shape[0]), self.__round_to_slice_size(shape[1]))

    def get_labeled_data(self, img_data, slice_step):
        x_img = cv2.imread(img_data[0], cv2.IMREAD_GRAYSCALE)
        y_img = cv2.imread(img_data[1], cv2.IMREAD_GRAYSCALE)
        x_img = self.__preprocess(x_img)
        y_img = self.normalize(y_img)
        samples = self.__take_samples(x_img, slice_step)
        labels = self.__label_samples(y_img, slice_step)
        return samples.assign(label=labels)

    @staticmethod
    def down_sample(df):
        minority_count = int(np.sum(df.loc[:, 'label']))
        minority_df = df[df.label == 1]
        majority_df = df[df.label == 0]
        majority_df = resample(majority_df, replace=False, n_samples=minority_count)
        return pd.concat([minority_df, majority_df])

