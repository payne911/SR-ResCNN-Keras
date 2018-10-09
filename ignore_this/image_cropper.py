import skimage.io
import matplotlib.pyplot as plt

from constants import tests_path


image = skimage.io.imread(tests_path + "2.png")

input_width  = 128
input_height = 128


# adapted from: https://stackoverflow.com/a/52463034/9768291
def seq_crop(img):
    """
    To crop the whole image in a list of sub-images of the same size.
    Size comes from "input_" variables in the 'constants' (Evaluation).
    Padding with 0 the Bottom and Right image.

    :param img: input image
    :return: list of sub-images with defined size
    """
    print("Shape of input image to crop:", img.shape[1], img.shape[0])
    width_shape = ceildiv(img.shape[1], input_width)
    height_shape = ceildiv(img.shape[0], input_height)
    sub_images = []  # will contain all the cropped sub-parts of the image

    for j in range(height_shape):
        horizontal = []
        for i in range(width_shape):
            horizontal.append(crop_precise(img, i*input_width, j*input_height, input_width, input_height))
        sub_images.append(horizontal)

    return sub_images


def crop_precise(img, coord_x, coord_y, width_length, height_length):
    """
    To crop a precise portion of an image.
    When trying to crop outside of the boundaries, the input to padded with zeros.

    :param img: image to crop
    :param coord_x: width coordinate (top left point)
    :param coord_y: height coordinate (top left point)
    :param width_length: width of the cropped portion starting from coord_x
    :param height_length: height of the cropped portion starting from coord_y
    :return: the cropped part of the image
    """
    tmp_img = img[coord_y:coord_y + height_length, coord_x:coord_x + width_length]

    return tmp_img


# from  https://stackoverflow.com/a/17511341/9768291
def ceildiv(a, b):
    """
    To get the ceiling of a division
    :param a:
    :param b:
    :return:
    """
    return -(-a // b)


print("Warning: numpy puts HEIGHT before WIDTH. Results shown reverse this and place WIDTH first.")
result = seq_crop(image)

plt.figure(figsize=(20, 20))
plt.suptitle("Crops")

tmp = 1
tmp_height = len(result)
tmp_width = len(result[0])
print("Shape of the list of crops: (", tmp_width, ",", tmp_height, ")")

for i in range(tmp_height):
    for j in range(tmp_width):
        plt.subplot(tmp_height, tmp_width, tmp)
        plt.imshow(result[i][j], cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
        tmp += 1

plt.show()
