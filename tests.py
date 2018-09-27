import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
#from PIL import Image

import utils
from constants import img_width
from constants import img_height
from constants import scale_fact
from constants import batch_size
from constants import verbosity
from constants import save_dir
from constants import weights
from constants import model_name


def test(model):
    x_test, y_test = extract_tests()

    evaluate(model, x_test, y_test)
    predict(model, x_test, y_test)


def extract_tests():
    print("Starting tests. Extracting images to feed.")
    x = []
    y = []

    # Extracting the benchmark images (HR)
    y_test1 = skimage.io.imread("pictures/final_tests/0764.png")
    y_test1 = crop_center(y_test1, img_width, img_height)
    y.append(y_test1)
    y_test2 = skimage.io.imread("pictures/final_tests/0774.png")
    y_test2 = crop_center(y_test2, img_width, img_height)
    y.append(y_test2)

    # Extracting unknown disturbance tests
    x_test1 = skimage.io.imread("pictures/final_tests/0764x4.png")
    x_test1 = crop_center(x_test1, img_width // scale_fact, img_height // scale_fact)
    x.append(x_test1)
    x_test2 = skimage.io.imread("pictures/final_tests/0774x4.png")
    x_test2 = crop_center(x_test2, img_width // scale_fact, img_height // scale_fact)
    x.append(x_test2)

    # Bicubic (?) downscale tests  TODO: verify the 'bicubic' claim
    x_test3 = utils.single_downscale(y_test1)
    x.append(x_test3)
    x_test4 = utils.single_downscale(y_test2)
    x.append(x_test4)
    # Because we're using the same ultimate goal
    y.append(y_test1)
    y.append(y_test2)

    return np.array(x), np.array(y)


def evaluate(model, x_test, y_test):
    print("Starting evaluation.")

    test_loss = model.evaluate(x_test,
                               y_test,
                               batch_size=batch_size,
                               verbose=verbosity)

    print('[evaluate] Test loss:', test_loss)
    #print('[evaluate] Test accuracy:', test_acc)

    # score = model.evaluate(x_test, y_test, verbose=False)
    # model.metrics_names
    # print('Test score: ', score[0])  # Loss on test
    # print('Test accuracy: ', score[1])


# Adapted from: https://stackoverflow.com/a/39382475/9768291
def crop_center(img, crop_x, crop_y):
    y, x, _ = img.shape
    start_x = x//2-(crop_x // 2)
    start_y = y//2-(crop_y // 2)

    cropped_img = img[start_y:start_y + crop_y, start_x:start_x + crop_x]

    return utils.float_im(cropped_img)


def predict(model, x_test, y_test):
    print("Starting predictions.")

    # # Trying to make predictions on a bunch of images (works in batches)
    # predictions = model.predict(images)

    # "model.predict" works in batches, so extracting a single prediction (for memory reasons):
    x_test1 = x_test[0]
    x_test2 = x_test[1]
    x_test3 = x_test[2]
    x_test4 = x_test[3]
    # We don't need the 2 other "y_test" since they are duplicates
    y_test1 = y_test[0]
    y_test2 = y_test[1]

    # Extracting predictions
    predictions = []
    input_img = (np.expand_dims(x_test1, 0))    # Add the image to a batch where it's the only member
    prediction1 = model.predict(input_img)[0]   # returns a list of lists, one for each image in the batch of data
    predictions.append(prediction1)
    input_img = (np.expand_dims(x_test2, 0))
    prediction2 = model.predict(input_img)[0]
    predictions.append(prediction2)
    input_img = (np.expand_dims(x_test3, 0))
    prediction3 = model.predict(input_img)[0]
    predictions.append(prediction3)
    input_img = (np.expand_dims(x_test4, 0))
    prediction4 = model.predict(input_img)[0]
    predictions.append(prediction4)

    # # TODO: Saving predictions
    # i = 0
    # save_path = "pictures/final_tests/predictions/"
    # print("Saving the 4 outputs as images")
    # for pred in predictions:
    #     utils.save_np_img(pred, save_path, str(i) + ".png")
    #     i += 1


    # # Taking the BICUBIC enlargment     TODO: figure out without taking the file from path again
    # bic1 = Image.open(data_path + '11.jpg').thumbnail((img_width, img_height), Image.BICUBIC)
    # bic2 = Image.open(data_path + '12.jpg').thumbnail((img_width, img_height), Image.BICUBIC)

    # TODO: Figure out how to show images to scale
    plt.figure(figsize=(12, 12))
    plt.suptitle("Results")

    # TODO: Turn into a `Class(4, 4, img, "title")`
    # input image
    plt.subplot(4, 4, 1)
    plt.title("Input: 128x128")
    plt.imshow(x_test1, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(4, 4, 2)
    plt.title("Input: 128x128")
    plt.imshow(x_test2, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(4, 4, 3)
    plt.title("Bicubic: 128x128")
    plt.imshow(x_test3, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(4, 4, 4)
    plt.title("Bicubic: 128x128")
    plt.imshow(x_test4, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    # bicubic enlargment  TODO: remove duplicate (input img) placeholder
    plt.subplot(4, 4, 5)
    plt.title("WIP (ignore)")
    plt.imshow(x_test1, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(4, 4, 6)
    plt.title("WIP (ignore)")
    plt.imshow(x_test2, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(4, 4, 7)
    plt.title("WIP (ignore)")
    plt.imshow(x_test3, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(4, 4, 8)
    plt.title("WIP (ignore)")
    plt.imshow(x_test4, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    # predicted image (x4 through network)
    plt.subplot(4, 4, 9)
    plt.title("Output: 512x512")
    plt.imshow(prediction1, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(4, 4, 10)
    plt.title("Output: 512x512")
    plt.imshow(prediction2, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(4, 4, 11)
    plt.title("Output: 512x512")
    plt.imshow(prediction3, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(4, 4, 12)
    plt.title("Output: 512x512")
    plt.imshow(prediction4, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    # initial image (HR)
    plt.subplot(4, 4, 13)
    plt.title("HR version: 512x512")
    plt.imshow(y_test1, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(4, 4, 14)
    plt.title("HR version: 512x512")
    plt.imshow(y_test2, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(4, 4, 15)
    plt.title("HR version: 512x512")
    plt.imshow(y_test1, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(4, 4, 16)
    plt.title("HR version: 512x512")
    plt.imshow(y_test2, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    plt.show()

    # https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.savefig
    # plt.savefig('pictures/final_tests/predictions/results.png', frameon=True) TODO: not working (white image)

    # Showing output vs expected image
    plt.figure(figsize=(20, 20))
    plt.suptitle("Results")
    # input image
    plt.subplot(1, 3, 1)
    plt.title("Input: 128x128")
    plt.imshow(x_test3, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(1, 3, 2)
    plt.title("Output: 512x512")
    plt.imshow(prediction3, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(1, 3, 3)
    plt.title("HR version: 512x512")
    plt.imshow(y_test1, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    plt.show()

    prompt_model_save(model, model_name)


def prompt_model_save(model, name):
    save_path = save_dir + '/' + name
    save_bool = input("Save weights of this model (y/n) ?")
    if save_bool == "y":
        model.save(save_path)
        del model  # deletes the existing model
        # model.save_weights('save/model_weights.h5')
