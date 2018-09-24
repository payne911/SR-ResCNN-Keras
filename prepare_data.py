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


def float_value_img(img):
    return np.divide(img, 255.)


# adapted from: https://stackoverflow.com/a/52463034/9768291
def random_crop(images):
    print("Started randomly cropping HR images")
    crop_h, crop_w = img_width, img_height
    y = []
    i = 0
    crops_per_image = 5
    for img in images:
        for _ in range(crops_per_image):
            if (img.shape[0] >= crop_h) and (img.shape[1] >= crop_w):
                # Cropping a random part of the image
                rand_h = np.random.randint(0, img.shape[0]-crop_h)
                rand_w = np.random.randint(0, img.shape[1]-crop_w)
                tmp_img = img[rand_h:rand_h+crop_h, rand_w:rand_w+crop_w]
                # TODO: switch from RGB channels to CbCrY
                # Saving the image
                save_np_img(tmp_img, "./pictures/HR/test_outputs/512/", str(i) + ".png")
                tmp_img = float_value_img(tmp_img)  # From [0,255] to [0.,1.]
                y.append(tmp_img)
                # Augmenting the image  TODO: look into integrating "imgaug" library
                if(augment_im):
                    initial_image = tmp_img
                    # Vertical axis flip
                    i += 1
                    tmp_img = np.fliplr(tmp_img)
                    save_np_img(tmp_img, "./pictures/HR/test_outputs/512/", str(i) + ".png")
                    tmp_img = float_value_img(tmp_img)
                    y.append(tmp_img)

                    # Horizontal axis flip of second image
                    i += 1
                    tmp_img = np.flipud(tmp_img)
                    save_np_img(tmp_img, "./pictures/HR/test_outputs/512/", str(i) + ".png")
                    tmp_img = float_value_img(tmp_img)
                    y.append(tmp_img)

                    # Horizontal axis flip of first image
                    i += 1
                    tmp_img = np.flipud(initial_image)
                    save_np_img(tmp_img, "./pictures/HR/test_outputs/512/", str(i) + ".png")
                    tmp_img = float_value_img(tmp_img)
                    y.append(tmp_img)

                # TODO: downsize from 512 to 128
                i += 1
                downscale(y)

            else:
                continue
    y = np.array(y)
    return y  # TODO: return (x, y) ?


def downscale(y):
    x = []
    # downsizing by scale of 4
    for i in range(len(y)):
        scaled_img = skimage.transform.resize(
            y[i],
            (img_width // scale_fact, img_height // scale_fact),
            mode='reflect',
            anti_aliasing=True)
        print(scaled_img)
        print(type(scaled_img))
        save_np_img(scaled_img, "./pictures/HR/test_outputs/128/", str(i) + ".png")
        tmp_img = float_value_img(tmp_img)
        x.append(scaled_img)

    return np.array(x)


if __name__ == '__main__':
    # Reading all images from folder
    images = [skimage.io.imread(path) for path in glob.glob('pictures/HR/*.png')]
    # Saving images after randomly cropping parts
    augs = random_crop(images)
    # TODO: modify `.fit` to use only (x,y) and set a value for validation ratio
