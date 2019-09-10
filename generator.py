import numpy as np
import glob
import os

import skimage.io

import math
from scipy.ndimage import sobel
from scipy.stats import truncnorm

from constants import img_width, img_height,\
    augment_img, crops_p_img, batch_size
from utils import float_im, single_downscale


class ImgDataGenerator:  # todo: extend keras.utils.Sequence
    def __init__(self,
                 path,                  # folder used by the generator to take images from
                 validation_split=0.0,  # percentage of the folder that will be split between validation and training set
                 nb_samples=0,          # number of images in the folder
                 random_samples=True):  # if 'False', will use a Sobel Filter and gaussian distribution to extract images
        if not os.path.isdir(path):
            raise Exception("The directory path for the Generator doesn't exist.")
        if validation_split < 0.0 or validation_split > 1.0:
            raise Exception("validation_split must be ranged inclusively between 0 and 1.")
        if validation_split > 0.0 and nb_samples <= 0:
            raise Exception("You must specify a positive amount of samples.")
        if not random_samples:
            self.sobel_sampler = SortedSobelImageProcessor(img_width, crops_p_img, 0.4)
        self.random_samples = random_samples
        self.split = validation_split
        self.path = path
        self.nb_samples = nb_samples

    def get_all_generators(self):
        """
        todo: use "multi_fold" generators to expose a bigger part of the dataset
        :return: Both the Training and the Validation generators.
        """
        return self.get_training_generator(), self.get_validation_generator()

    def get_full_generator(self):
        """
        :return: The generator that extracts all the data from the specified folder.
        """
        if not self.split == 0.0:
            raise Exception("Use the specific getters when a non-default validation_split value is provided.")

        i = 0
        while True:
            pattern_path = self.__get_pattern_match()
            y = []
            amount = 0
            for img_path in glob.iglob(pattern_path):  # avoids loading whole dataset in RAM
                i += 1
                amount = self.__extract(amount, img_path, y)  # extract multiple crops from a single image

                if amount % batch_size == 0:  # yielding a single batch
                    amount = 0
                    yield self.__extract_yield(y)
                    y = []

            if y:  # if iglob finished but didn't yield because of modulo not reached
                yield self.__extract_yield(y)
            i = 0

    def get_training_generator(self):
        """
        :return: The generator that only extracts from the Training set.
        """
        if self.split == 0.0:
            raise Exception("You must have specified a non-default validation_split value to use this.")

        exclusion_index = self.nb_samples - int(self.split * self.nb_samples)
        i = 0
        while True:
            pattern_path = self.__get_pattern_match()
            y = []
            amount = 0
            for img_path in glob.iglob(pattern_path):  # avoids loading whole dataset in RAM
                i += 1
                if i > exclusion_index:  # excluding validation set
                    continue

                amount = self.__extract(amount, img_path, y)  # extract multiple crops from a single image
                print("treated img: " + img_path)
                if amount % batch_size == 0:  # yielding a single batch
                    amount = 0
                    yield self.__extract_yield(y)
                    y = []

            if y:  # if iglob finished but didn't yield because of modulo not reached
                yield self.__extract_yield(y)
            i = 0

    def get_validation_generator(self):
        """
        :return: The generator that only extracts from the Validation set.
        """
        if self.split == 0.0:
            raise Exception("You must have specified a non-default validation_split value to use this.")

        exclusion_index = self.nb_samples - int(self.split * self.nb_samples)
        i = 0
        while True:
            pattern_path = self.__get_pattern_match()
            y = []
            amount = 0
            for img_path in glob.iglob(pattern_path):  # avoids loading whole dataset in RAM
                i += 1
                if i <= exclusion_index:  # excluding training set
                    continue

                amount = self.__extract(amount, img_path, y)  # extract multiple crops from a single image

                if amount % batch_size == 0:  # yielding a single batch
                    amount = 0
                    yield self.__extract_yield(y)
                    y = []

            if y:  # if iglob finished but didn't yield because of modulo not reached
                yield self.__extract_yield(y)
            i = 0

    def __extract(self, amount, img_path, y):
        img = skimage.io.imread(img_path)

        if self.random_samples:
            y.extend(self.__random_crops(img))  # random crops
        else:
            y.extend(self.__sobel_sample(img))  # gaussian crops

        # Augmenting the image  TODO: look into integrating "imgaug" library
        if augment_img:
            z = []
            for img in y:
                z.append(np.fliplr(img))  # symmetry on 'y' axis
            y.extend(z)
        return amount+1

    def __downscale(self, images):
        x = []
        for img in images:
            x.append(single_downscale(img, img_width, img_height))
        return np.array(x)

    def __get_pattern_match(self):
        return self.path + '*.png'

    def __extract_yield(self, y):
        y = np.array(y)
        x = self.__downscale(y)  # downscaling all those images
        return {'input': x}, {'output': y}

    # adapted from: https://stackoverflow.com/a/52463034/9768291
    def __random_crops(self, img):
        crop_h, crop_w = img_width, img_height
        y = []

        if (img.shape[0] >= crop_h) and (img.shape[1] >= crop_w):
            # img = __rgb2ycbcr(img)  # TODO: switch from RGB channels to CbCrY
            for _ in range(crops_p_img):
                # Cropping a random part of the image
                rand_h = np.random.randint(0, img.shape[0] - crop_h)
                rand_w = np.random.randint(0, img.shape[1] - crop_w)
                tmp_img = img[rand_h:rand_h + crop_h, rand_w:rand_w + crop_w]

                y.append(float_im(tmp_img))  # From [0,255] to [0.,1.]

        return y

    def __sobel_sample(self, img):
        return self.sobel_sampler.sobel_sample(img)

    # from: https://stackoverflow.com/a/34913974/9768291
    def __rgb2ycbcr(self, im):
        # TODO: fix
        xform = np.array([[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]])
        ycbcr = im.dot(xform.T)
        ycbcr[:, :, [1, 2]] += 128
        return np.uint8(ycbcr)


class SortedSobelImageProcessor:
    def __init__(self,
                 crop_size,  # Size of the crops
                 crop_amnt,  # The amount of images extracted
                 sigma):     # The sampling distribution value
        self.crop_size = crop_size
        self.crop_amnt = crop_amnt
        self.sigma = sigma

    def sobel_sample(self, img):
        """
        Uses the sigma value to sample the input image path.
        A lower sigma value (like 0.2) will result in more complex images being returned.
        The sigma has no maximum value.
        :param img: an image.
        :return: "batch_size" amount of images extracted from the input.
        """
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

                imgcrop = float_im(img[ystart:yend, xstart:xend, :])  # From [0,255] to [0.,1.]
                images.append(imgcrop)

                imgsobel = np.sqrt((sobel(imgcrop, axis=-3) ** 2) + (sobel(imgcrop, axis=-2) ** 2)).mean()
                images_gradient.append(imgsobel)

        images = np.array(images)
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
            np.round(np.abs(truncnorm.rvs(a=0, b=1.0 / self.sigma, scale=images_max * self.sigma, size=self.crop_amnt))), 0,
            images_max).astype(np.int)
        return images_sorted[randomidx]
