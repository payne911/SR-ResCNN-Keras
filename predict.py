import argparse
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

from keras.models import load_model

from constants import verbosity
from constants import save_dir
from constants import model_name
from constants import crops_p_img
from constants import tests_path
from constants import scale_fact
from constants import input_width
from constants import input_height
from utils import float_im
from utils import crop_center


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-a', '--amount', type=int, default=crops_p_img,
                    help='how many (cropped to 128x128) samples to predict from within the image')
parser.add_argument('image', type=str,
                    help='image name (example: "bird.png") that must be inside the "./input/" folder')
parser.add_argument('-m', '--model', type=str, default=model_name,
                    help='model name (in the "./save/" folder), followed by ".h5"')
parser.add_argument('-r', '--random', action="store_true",  # if var is in args, set to TRUE, else, set to FALSE
                    help='flag that will select a random 128x128 area in the input image instead of the center')
parser.add_argument('-f', '--full', action="store_true",  # if var is in args, set to TRUE, else, set to FALSE
                    help='(WIP) flag that will get the whole image to be processed by the network')

args = parser.parse_args()


# TODO: redo picture 1 and 2 in "input/downscaled"


def predict(args):
    model = load_model(save_dir + '/' + args.model)

    image = skimage.io.imread(tests_path + args.image)

    if (image.shape[0] == input_width) or args.full:
        args.amount = 1

    predictions = []
    images = []

    if args.full:
        crops = seq_crop(image)  # crops into multiple sub-parts the image based on 'input_' constants

        for i in range(len(crops)):  # amount of vertical crops
            for j in range(len(crops[0])):  # amount of horizontal crops
                current_image = crops[i][j]
                images.append(current_image)

        print("Moving on to predictions. Amount:", len(images))

        for p in range(len(images)):
            if p%3 == 0 and verbosity == 2:
                print("--prediction #", p)

            # Hack because GPU can only handle one image at a time
            input_img = (np.expand_dims(images[p], 0))       # Add the image to a batch where it's the only member
            predictions.append(model.predict(input_img)[0])  # returns a list of lists, one for each image in the batch
    else:  # if the "-f" flag isn't set
        for i in range(args.amount):
            # Cropping to fit input size
            if (args.random or args.amount > 1) and image.shape[0] > input_width:
                images.append(random_crop(image))
            else:
                images.append(crop_center(image, input_width, input_height))

            # Hack because GPU can only handle one image at a time
            input_img = (np.expand_dims(images[i], 0))       # Add the image to a batch where it's the only member
            predictions.append(model.predict(input_img)[0])  # returns a list of lists, one for each image in the batch

    # Comparing originals with the SR-versions
    for i in range(len(predictions)):
        show_pred_output(images[i], predictions[i])

    return predictions, image


# adapted from: https://stackoverflow.com/a/52463034/9768291
def random_crop(img):
    crop_h, crop_w = input_width, input_height
    print("Shape of input image to crop:", img.shape[0], img.shape[1])

    if (img.shape[0] >= crop_h) and (img.shape[1] >= crop_w):
        # Cropping a random part of the image
        rand_h = np.random.randint(0, img.shape[0]-crop_h)
        rand_w = np.random.randint(0, img.shape[1]-crop_w)
        print("Random position for the crop:", rand_h, rand_w)
        tmp_img = img[rand_h:rand_h+crop_h, rand_w:rand_w+crop_w]

        new_img = float_im(tmp_img)  # From [0,255] to [0.,1.]
    else:
        return img

    return new_img


def show_pred_output(input, pred):
    plt.figure(figsize=(20, 20))
    plt.suptitle("Results")

    plt.subplot(1, 2, 1)
    plt.title("Input: " + str(input_width // scale_fact) + "x" + str(input_height // scale_fact))
    plt.imshow(input, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    plt.subplot(1, 2, 2)
    plt.title("Output: " + str(input_width) + "x" + str(input_height))
    plt.imshow(pred, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    plt.show()


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

    return float_im(tmp_img)  # From [0,255] to [0.,1.]


# from  https://stackoverflow.com/a/17511341/9768291
def ceildiv(a, b):
    """
    To get the ceiling of a division
    :param a:
    :param b:
    :return:
    """
    return -(-a // b)


def reconstruct(predictions):
    width_length = 0
    height_length = 0

    # TODO: properly extract the size of the full image
    for width_imgs in predictions[0]:
        width_length += width_imgs.shape[1]
    for height_imgs in predictions:
        height_length += height_imgs[0].shape[0]
    print(width_length, height_length)

    full_image = np.empty(shape=(height_length, width_length))
    print(full_image.shape)

    # TODO: properly merge the crops back into a single image
    for height in range(len(predictions[0])):
        for width in range(len(predictions)):
            # concatenate here
            print(height, width)

    return full_image


if __name__ == '__main__':
    print("   -  ", args)
    preds, original = predict(args)  # returns the predictions along with the original

    # # TODO: reconstruct image
    # enhanced = reconstruct(preds)  # reconstructs the enhanced image from predictions
    # show_pred_output(original, enhanced)
