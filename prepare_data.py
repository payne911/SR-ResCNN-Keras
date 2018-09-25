import numpy as np
import glob

from PIL import Image
import skimage
from skimage import io
from skimage import transform

from constants import y_data_path
from constants import img_width
from constants import img_height
from constants import scale_fact
from constants import augment_im
from model import setUpModel


def save_np_img(np_img, path, name):
    """
    To save the image.
    :param np_img: numpy_array type image
    :param path: string type of the existing path where to save the image
    :param name: string type that includes the format (ex:"bob.png")
    :return: numpy array divided by 255 (float values between 0 and 1)
    """

    assert isinstance(path, str), 'Path of wrong type! (Must be String)'
    assert isinstance(name, str), 'Name of wrong type! (Must be String)'

    im = Image.fromarray(np_img)
    im.save(path + name)

    return np_img


def float_im(img):
    return np.divide(img, 255.)

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
    i = 0
    crops_per_image = 5

    for img in images:
        if (img.shape[0] >= crop_h) and (img.shape[1] >= crop_w):
            #img = rgb2ycbcr(img)  # TODO: switch from RGB channels to CbCrY
            for _ in range(crops_per_image):
                # Cropping a random part of the image
                rand_h = np.random.randint(0, img.shape[0]-crop_h)
                rand_w = np.random.randint(0, img.shape[1]-crop_w)
                tmp_img = img[rand_h:rand_h+crop_h, rand_w:rand_w+crop_w]

                # Saving the images
                initial_image = save_np_img(tmp_img, y_data_path, str(i) + ".png")
                y.append(float_im(initial_image))  # From [0,255] to [0.,1.]

                # Augmenting the image  TODO: look into integrating "imgaug" library
                if(augment_im):
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


def downscale(images):

    x = []

    for img in images:
        scaled_img = skimage.transform.resize(
            img,
            (img_width // scale_fact, img_height // scale_fact),
            mode='reflect',
            anti_aliasing=True)
        print(img)
        #x.append(scaled_img)

    return np.array(x)


if __name__ == '__main__':
    # Reading all images from folder
    images = [skimage.io.imread(path) for path in glob.glob('pictures/HR/*.png')]
    # Saving images after randomly cropping parts
    y_train = random_crop(images)
    x_train = downscale(y_train)
    print(x_train.shape)

    #setUpModel(y_train)
    # TODO: modify `.fit` to use only (x,y) and set a value for validation ratio
