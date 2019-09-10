import skimage.io
import numpy as np
import os

from PIL import Image
import requests
from io import BytesIO
import math
from scipy.ndimage import sobel
from scipy.stats import truncnorm


class SortedSobelImageProcessor:
    def __init__(self,
                 crop_size,        # Size of the crops (a single integer)
                 batch_size,       # The amount of images extracted
                 sigma,            # The sampling distribution value
                 internet=False):  # 'True' if the paths will be coming from the Internet
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.sigma = sigma
        self.internet = internet

    def sobel_sample(self, path):
        """
        Uses the sigma value to sample the input image path.
        A lower sigma value (like 0.2) will result in more complex images being returned.
        The sigma has no maximum value.
        :param path: path to an image.
        :return: "batch_size" amount of images extracted from the input.
        """
        if self.internet:  # web
            img = np.array(Image.open(BytesIO(requests.get(path).content)))
        elif os.path.isfile(path):  # file
            img = skimage.io.imread(path)
        else:
            raise Exception("Can't find the file.")

        images, images_gradient = self.__crop_image(img)             # Crop the image
        images_sorted = self.__sort_images(images, images_gradient)  # Sort the crops
        return self.__sample_gaussian(images_sorted)                 # Sampled gaussian distribution

    def __crop_image(self, img):
        """
        Splits the image, extracting the Sobel component (edge detection) at the same time.
        :param img: input image to be split and treated.
        :return: list of sub-images, and a list of their corresponding Sobel component's value.
        """
        images = []
        images_gradient = []

        for y in range(math.floor(float(img.shape[-3]) / self.crop_size)):
            ystart = y * self.crop_size
            yend = ystart + self.crop_size

            for x in range(math.floor(float(img.shape[-2]) / self.crop_size)):
                xstart = x * self.crop_size
                xend = xstart + self.crop_size

                imgcrop = img[ystart:yend, xstart:xend, :]
                images.append(imgcrop)

                imgsobel = np.sqrt((sobel(imgcrop / 255, axis=-3) ** 2) + (sobel(imgcrop / 255, axis=-2) ** 2)).mean()
                images_gradient.append(imgsobel)

        images = np.array(images, dtype='uint8')
        images_gradient = np.array(images_gradient)

        return images, images_gradient

    def __sort_images(self, images, images_gradient):
        """
        Sorts the cropped images by their gradient amplitude.
        :param images: first output from the "crop_image" function.
        :param images_gradient: second output from the "crop_image" function.
        :return: the sorted input arrays.
        """
        im_argsort = np.flip(np.argsort(images_gradient, kind='stable'))
        return images[im_argsort]

    def __sample_gaussian(self, images_sorted):
        """
        Selects random images with gaussian distribution with replacement.
        :param images_sorted: sorted numpy array of cropped images (first output from "sort_images" function).
        :return: a "batch_size" amount of randomly sampled images, according to the sigma parameter of the class.
        """
        images_max = len(images_sorted) - 1
        randomidx = np.clip(
            np.round(np.abs(truncnorm.rvs(a=0, b=1.0 / self.sigma, scale=images_max * self.sigma, size=self.batch_size))), 0,
            len(images_sorted) - 1).astype(np.int)
        return images_sorted[randomidx]


test = SortedSobelImageProcessor(128, 10, 0.2, True)
print(len(test.sobel_sample('http://vllab.ucmerced.edu/wlai24/LapSRN/results/Urban100/x4/img_078_x4_GT.png')))
