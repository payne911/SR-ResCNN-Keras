import numpy as np
import glob

import skimage.io

from constants import img_width, img_height,\
    augment_img, crops_p_img, batch_size
from utils import float_im, single_downscale


class ImgDataGenerator:
    def __init__(self,
                 path,
                 validation_split=0.0,
                 nb_samples=0):
        if validation_split < 0.0 or validation_split > 1.0:
            raise Exception("validation_split must be ranged inclusively between 0 and 1.")
        if validation_split > 0.0 and nb_samples <= 0:
            raise Exception("You must specify a positive amount of samples.")
        self.split = validation_split
        self.path = path
        self.nb_samples = nb_samples

    def get_all_generators(self):
        return self.get_training_generator(), self.get_validation_generator()

    def get_full_generator(self):
        if not self.split == 0.0:
            raise Exception("Use the specific getters when a non-default"
                            "validation_split value is provided.")

        i = 0
        while True:
            pattern_path = self.__get_pattern_match()
            y = []
            amount = 0
            for img_path in glob.iglob(pattern_path):  # avoids loading whole dataset in RAM
                img = skimage.io.imread(img_path)
                amount += 1
                i += 1
                y.extend(self.__random_crops(img))  # adding multiple crops out of a single image

                if amount % batch_size == 0:  # yielding a single batch
                    amount = 0
                    yield self.__extract_yield(y)
                    y = []

            if y:  # if iglob finished but didn't yield because of modulo not reached
                print("Yielding the final images that were left behind.")
                yield self.__extract_yield(y)
            i = 0

    def get_training_generator(self):
        if self.split == 0.0:
            raise Exception("You must have specified a non-default"
                            "validation_split value to use this.")

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

                img = skimage.io.imread(img_path)
                amount += 1
                y.extend(self.__random_crops(img))  # adding multiple crops out of a single image

                if amount % batch_size == 0:  # yielding a single batch
                    amount = 0
                    yield self.__extract_yield(y)
                    y = []

            if y:  # if iglob finished but didn't yield because of modulo not reached
                yield self.__extract_yield(y)
            i = 0

    def get_validation_generator(self):
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
                if i <= exclusion_index:  # excluding validation set
                    continue

                img = skimage.io.imread(img_path)
                amount += 1
                y.extend(self.__random_crops(img))  # adding multiple crops out of a single image

                if amount % batch_size == 0:  # yielding a single batch
                    amount = 0
                    yield self.__extract_yield(y)
                    y = []

            if y:  # if iglob finished but didn't yield because of modulo not reached
                yield self.__extract_yield(y)
            i = 0

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

                # Augmenting the image  TODO: look into integrating "imgaug" library
                if augment_img:
                    y.append(float_im(np.fliplr(tmp_img)))  # symmetry on 'y' axis

        return y

    # from: https://stackoverflow.com/a/34913974/9768291
    def __rgb2ycbcr(self, im):
        # TODO: fix
        xform = np.array([[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]])
        ycbcr = im.dot(xform.T)
        ycbcr[:, :, [1, 2]] += 128
        return np.uint8(ycbcr)
