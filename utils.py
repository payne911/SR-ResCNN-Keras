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
from constants import model_saved
from constants import model_name


def loadData():
    print("Loading data.")
    images = [float_im(skimage.io.imread(path)) for path in glob.glob(y_data_path + "*.png")]  # TODO: customize with command line
    setUpData(np.array(images))


def float_im(img):
    return np.divide(img, 255.)


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

    if model_saved:
        load_saved_model(model_name, x_train, y_train)
    else:
        setUpModel(x_train, y_train)









    ###########################
    #        DRAWINGS         #
    ###########################

# def plot_image(i, predictions_array, true_label, img):
#     predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.imshow(img, cmap=plt.cm.binary)
#
#     predicted_label = np.argmax(predictions_array)
#     if predicted_label == true_label:
#         color = 'blue'
#     else:
#         color = 'red'
#
#     plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                          100 * np.max(predictions_array),
#                                          class_names[true_label]),
#                color=color)
#
# def plot_value_array(i, predictions_array, true_label):
#     predictions_array, true_label = predictions_array[i], true_label[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     thisplot = plt.bar(range(10), predictions_array, color="#777777")
#     plt.ylim([0, 1])
#     predicted_label = np.argmax(predictions_array)
#     plt.xticks(range(10))  # adding the class-index below prediction graph
#
#     thisplot[predicted_label].set_color('red')
#     thisplot[true_label].set_color('blue')
#
# # def draw_prediction(index):
# #     plt.figure(figsize=(6, 3))
# #     plt.subplot(1, 2, 1)
# #     plot_image(index, predictions, test_labels, test_images)
# #     plt.subplot(1, 2, 2)
# #     plot_value_array(index, predictions, test_labels)
# #     plt.show()
# #
# # To draw a single prediction
# # draw_prediction(0)
# # draw_prediction(12)
#
# # Plot the first X test images, their predicted label, and the true label
# # Color correct predictions in blue, incorrect predictions in red
# num_rows = 5
# num_cols = 3
# num_images = num_rows * num_cols
# plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

# Adding a title to the plot
# plt.suptitle("Check it out!")

# for i in range(num_images):
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#     plot_image(i, predictions, test_labels, test_images)
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#     plot_value_array(i, predictions, test_labels)
# plt.show()
