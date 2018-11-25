import argparse
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

from keras.models import load_model

from constants import verbosity
from constants import save_dir
from constants import model_name
from constants import tests_path
from constants import input_width
from constants import input_height
from utils import float_im


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('image', type=str,
                    help='image name (example: "bird.png") that must be inside the "./input/" folder')
parser.add_argument('-m', '--model', type=str, default=model_name,
                    help='model name (in the "./save/" folder), followed by ".h5"')
parser.add_argument('-s', '--save', type=str, default='your_image.png',
                    help='the name of the saved image which will appear inside the "output" folder')

args = parser.parse_args()


# TODO: redo picture 1 and 2 in "input/downscaled"


def predict(args):
    model = load_model(save_dir + '/' + args.model)

    image = skimage.io.imread(tests_path + args.image)[:, :, :3]  # removing possible extra channels (Alpha)
    print("Image shape:", image.shape)

    predictions = []
    images = []

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

    return predictions, image, crops


def show_pred_output(input, pred):
    plt.figure(figsize=(20, 20))
    plt.suptitle("Results")

    plt.subplot(1, 2, 1)
    plt.title("Input : " + str(input.shape[1]) + "x" + str(input.shape[0]))
    plt.imshow(input, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    plt.subplot(1, 2, 2)
    plt.title("Output : " + str(pred.shape[1]) + "x" + str(pred.shape[0]))
    plt.imshow(pred, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    plt.show()


# adapted from  https://stackoverflow.com/a/52463034/9768291
def seq_crop(img):
    """
    To crop the whole image in a list of sub-images of the same size.
    Size comes from "input_" variables in the 'constants' (Evaluation).
    Padding with 0 the Bottom and Right image.

    :param img: input image
    :return: list of sub-images with defined size
    """
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
    return -(-a // b)


# adapted from  https://stackoverflow.com/a/52733370/9768291
def reconstruct(predictions, crops):

    # unflatten predictions
    def nest(data, template):
        data = iter(data)
        return [[next(data) for _ in row] for row in template]

    if len(crops) != 0:
        predictions = nest(predictions, crops)

    H = np.cumsum([x[0].shape[0] for x in predictions])
    W = np.cumsum([x.shape[1] for x in predictions[0]])
    D = predictions[0][0]
    recon = np.empty((H[-1], W[-1], D.shape[2]), D.dtype)
    for rd, rs in zip(np.split(recon, H[:-1], 0), predictions):
        for d, s in zip(np.split(rd, W[:-1], 1), rs):
            d[...] = s
    return recon


if __name__ == '__main__':
    print("   -  ", args)

    preds, original, crops = predict(args)  # returns the predictions along with the original
    enhanced = reconstruct(preds, crops)    # reconstructs the enhanced image from predictions

    plt.imsave('output/' + args.save, enhanced, cmap=plt.cm.gray)

    show_pred_output(original, enhanced)
