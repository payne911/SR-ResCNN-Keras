import argparse
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

from keras.models import load_model
from keras.optimizers import Adam
from keras.optimizers import Adadelta

from constants import save_dir
from constants import model_name
from constants import crops_p_img
from constants import tests_path
from constants import img_height
from constants import img_width
from constants import scale_fact
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

    # Setting up the proper optimizer       TODO: needed?
    if args.model == "my_full_model.h5":
        optimizer = Adadelta(lr=1.0,
                             rho=0.95,
                             epsilon=None,
                             decay=0.0)
    else:
        optimizer = Adam(lr=0.001,
                         beta_1=0.9,
                         beta_2=0.999,
                         epsilon=None,
                         decay=0.0,
                         amsgrad=False)

    model.compile(optimizer=optimizer,
                  loss='mean_squared_error')

    image = skimage.io.imread(tests_path + args.image)

    if image.shape[0] == 128:
        args.amount = 1

    predictions = []
    images = []

    # TODO: integrate FULL IMAGE
    # if args.full:
    #     images.append(image)
    #     # Hack because GPU can only handle one image at a time
    #     input_img = (np.expand_dims(images[0], 0))  # Add the image to a batch where it's the only member
    #     predictions.append(model.predict(input_img)[0])  # returns a list of lists, one for each image in the batch
    # else:
    if True:
        for i in range(args.amount):
            # Cropping to fit input size
            if (args.random or args.amount > 1) and image.shape[0] > 128:
                images.append(random_crop(image))
            else:
                images.append(crop_center(image, img_width//scale_fact, img_height//scale_fact))

            input_img = (np.expand_dims(images[i], 0))
            predictions.append(model.predict(input_img)[0])

    for i in range(len(predictions)):
        show_pred_output(images[i], predictions[i])


# adapted from: https://stackoverflow.com/a/52463034/9768291
def random_crop(img):
    crop_h, crop_w = img_width//scale_fact, img_height//scale_fact
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
    plt.title("Input: 128x128")
    plt.imshow(input, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    plt.subplot(1, 2, 2)
    plt.title("Output: 512x512")
    plt.imshow(pred, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    plt.show()


if __name__ == '__main__':
    print("   -  ", args)
    predict(args)
