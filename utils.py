import numpy as np
#import matplotlib.pyplot as plt

import skimage
from skimage import transform
from skimage import io
from PIL import Image
import glob

from constants import y_data_path
from constants import img_width
from constants import img_height
from constants import scale_fact

from model import setUpModel
from model import load_saved_model
from constants import load_model


def loadData():
    print("Loading data.")
    images = [float_im(skimage.io.imread(path)) for path in glob.glob(y_data_path + "*.png")]  # TODO: customize with command line

    print("Converting to Numpy Array.")
    setUpData(np.array(images))


def float_im(img):
    return np.divide(img, 255.)


# Adapted from: https://stackoverflow.com/a/39382475/9768291
def crop_center(img, crop_x, crop_y):
    y, x, _ = img.shape
    start_x = x//2-(crop_x // 2)
    start_y = y//2-(crop_y // 2)

    cropped_img = img[start_y:start_y + crop_y, start_x:start_x + crop_x]

    return float_im(cropped_img)


# TODO: provide some way of saving FLOAT images
def save_np_img(np_img, path, name):
    """
    To save the image.
    :param np_img: numpy_array type image
    :param path: string type of the existing path where to save the image
    :param name: string type that includes the format (ex:"bob.png")
    :return: numpy array
    """

    assert isinstance(path, str), 'Path of wrong type! (Must be String)'
    assert isinstance(name, str), 'Name of wrong type! (Must be String)'

    # TODO: To transform float-arrays into int-arrays (see https://stackoverflow.com/questions/52490653/saving-float-numpy-images)
    if type(np_img[0][0][0].item()) != int:
        np_img = np.multiply(np_img, 255).astype(int)
        # File "C:\Users\payne\Anaconda3\envs\ml-gpu\lib\site-packages\PIL\Image.py", line 2460, in fromarray
        #     mode, rawmode = _fromarray_typemap[typekey]
        # KeyError: ((1, 1, 3), '<i4')
        # File  "C:\Users\payne\Anaconda3\envs\ml-gpu\lib\site-packages\PIL\Image.py", line 2463, in fromarray
        #     raise TypeError("Cannot handle this data type")
        # TypeError: Cannot handle this data type

    im = Image.fromarray(np_img)
    im.save(path + name)

    return np_img


def downscale(images):
    print("Downscaling training set.")
    x = []

    for img in images:
        x.append(single_downscale(img))

    return np.array(x)


def single_downscale(img):
    """
    Downscales an image by the factor set in the 'constants'
    :param img: the image
    :return: returns a float-type numpy by default (values between 0 and 1)
    """
    # TODO: look into `skimage.transform.downscale_local_mean()`
    scaled_img = skimage.transform.resize(
        img,
        (img_width // scale_fact, img_height // scale_fact),
        mode='reflect',
        anti_aliasing=True)

    return scaled_img


# TODO: Use Keras "ImageDataGenerator" (https://stackoverflow.com/a/52462868/9768291) (https://stackoverflow.com/a/43382171/9768291)
def setUpData(y_train):
    print("Shape of the training set created:", y_train.shape)
    x_train = downscale(y_train)

    # TODO: Substract mean of all images

    # # Sanity check: display eight images
    # plt.figure(figsize=(10, 10))
    # for i in range(4):
    #     plt.subplot(3, 3, i + 1)
    #     plt.imshow(y_train[i], cmap=plt.cm.binary)
    # for i in range(4):
    #     plt.subplot(3, 3, i + 1 + 4)
    #     plt.imshow(x_train[i], cmap=plt.cm.binary)
    # plt.show()

    if load_model:
        load_saved_model(x_train, y_train)
    else:
        setUpModel(x_train, y_train)
