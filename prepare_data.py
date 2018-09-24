import numpy as np
import glob

from PIL import Image
import skimage
from skimage import io
from skimage import transform

from constants import data_path
from constants import img_width
from constants import img_height
from constants import scale_fact
from constants import augment_im


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
                rand_h = np.random.randint(0, img.shape[0]-crop_h)
                rand_w = np.random.randint(0, img.shape[1]-crop_w)
                tmp_img = img[rand_h:rand_h+crop_h, rand_w:rand_w+crop_w]
                # TODO: switch from RGB channels to CbCrY
                im = Image.fromarray(tmp_img)
                im.save("./pictures/HR/test_outputs/512/" + str(i) + ".png")
                y.append(np.divide(tmp_img, 255.))
                # if(augment_im) augment_img(y)     TODO: integrate
                # TODO: downsize from 512 to 128
                i += 1
            else:
                continue
    y = np.array(y)
    return y  # TODO: return (x, y) ?


# TODO: look into integrating "imgaug" library
def augment_img(dataToAugment):
    print("Starting to augment data")
    y = []

    # vertical axis flip         (-> x2)
    for i in range(len(dataToAugment)):
        y.append(np.fliplr(y[i]))
    # horizontal axis flip       (-> x2)
    for i in range(len(y)):
        y.append(np.flipud(y[i]))

    return np.array(y)


def downscale(y):
    x = []
    # downsizing by scale of 4
    for i in range(len(y)):
        x.append(skimage.transform.resize(
            y[i],
            (img_width / scale_fact, img_height / scale_fact),
            mode='reflect',
            anti_aliasing=True))

    return np.array(x)


if __name__ == '__main__':
    # Reading all images from folder
    images = [skimage.io.imread(path) for path in glob.glob('pictures/HR/*.png')]
    # Saving images after randomly cropping parts
    augs = random_crop(images)
    # TODO: modify `.fit` to use only (x,y) and set a value for validation ratio
