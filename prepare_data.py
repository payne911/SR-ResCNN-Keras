import numpy as np
import glob

import skimage
from skimage import io

from constants import y_data_path
from constants import img_width
from constants import img_height
from constants import augment_img
from constants import crops_p_img

from utils import setUpData
from utils import save_np_img
from utils import float_im


# from: https://stackoverflow.com/a/34913974/9768291
def rgb2ycbcr(im):
    # TODO: fix
    xform = np.array([[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)


# adapted from: https://stackoverflow.com/a/52463034/9768291
def random_crop(images):
    print("Started randomly cropping HR images")
    crop_h, crop_w = img_width, img_height
    y = []
    i = -1  # To begin with 0

    for img in images:
        if (img.shape[0] >= crop_h) and (img.shape[1] >= crop_w):
            print(i + 1, "images processed")
            #img = rgb2ycbcr(img)  # TODO: switch from RGB channels to CbCrY
            for _ in range(crops_p_img):
                # Cropping a random part of the image
                rand_h = np.random.randint(0, img.shape[0]-crop_h)
                rand_w = np.random.randint(0, img.shape[1]-crop_w)
                tmp_img = img[rand_h:rand_h+crop_h, rand_w:rand_w+crop_w]

                # Saving the images
                i += 1
                initial_image = save_np_img(tmp_img, y_data_path, str(i) + ".png")
                y.append(float_im(initial_image))  # From [0,255] to [0.,1.]

                # Augmenting the image  TODO: look into integrating "imgaug" library
                if augment_img:
                    # Vertical axis flip
                    i += 1
                    tmp_img = np.fliplr(initial_image)
                    y.append(float_im(save_np_img(tmp_img, y_data_path, str(i) + ".png")))

                    # Horizontal axis flip of tmp_img
                    i += 1
                    y.append(float_im(save_np_img(np.flipud(tmp_img), y_data_path, str(i) + ".png")))

                    # Horizontal axis flip of initial_image
                    i += 1
                    y.append(float_im(save_np_img(np.flipud(initial_image), y_data_path, str(i) + ".png")))
            else:
                continue
    return np.array(y)


def load_imgs(path):
    # Reading all images from folder
    images = [skimage.io.imread(path) for path in glob.glob(path)]

    # Saving images after randomly cropping parts
    y_train = random_crop(images)

    setUpData(y_train)
