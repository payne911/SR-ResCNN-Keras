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


def test(model):
    print("Starting tests. Extracting images to feed.")
    x_test, y_test = extract_tests()
    predict(model, x_test, y_test)
    evaluate(model, x_test, y_test)  # TODO: once evaluate is stable, put BEFORE 'predict' ?


def extract_tests():
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
    # TODO: evaluate separately?
    print(type(x_test))
    print(x_test.shape)
    print(type(y_test))
    print(y_test.shape)
    # ValueError: Error when checking model input: the list of Numpy arrays that you are passing to your model is not the size the model expected. Expected to see 1 array(s), but instead got the following list of 4 arrays: [array([[[1.        , 0.95686275, 0.71764706],
    test_loss, test_acc = model.evaluate(x_test,
                                         y_test,
                                         batch_size=batch_size,
                                         verbose=verbosity)
    print('[evaluate] Test loss:', test_loss)
    print('[evaluate] Test accuracy:', test_acc)

    # score = model.evaluate(x_test, y_test, verbose=False)
    # model.metrics_names
    # print('Test score: ', score[0])  # Loss on test
    # print('Test accuracy: ', score[1])


# Adapted from: https://stackoverflow.com/a/39382475/9768291
def crop_center(img, crop_x, crop_y):
    y, x, _ = img.shape
    start_x = x//2-(crop_x // 2)
    start_y = y//2-(crop_y // 2)
    return utils.float_im(img[start_y:start_y + crop_y, start_x:start_x + crop_x])


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
    print("Predicting on 1st input")
    input_img = (np.expand_dims(x_test1, 0))    # Add the image to a batch where it's the only member
    prediction1 = model.predict(input_img)[0]   # returns a list of lists, one for each image in the batch of data
    predictions.append(prediction1)
    print("Predicting on 2nd input")
    input_img = (np.expand_dims(x_test2, 0))
    prediction2 = model.predict(input_img)[0]
    predictions.append(prediction2)
    print("Predicting on 3rd input")
    input_img = (np.expand_dims(x_test3, 0))
    prediction3 = model.predict(input_img)[0]
    predictions.append(prediction3)
    print("Predicting on 4th input")
    input_img = (np.expand_dims(x_test4, 0))
    prediction4 = model.predict(input_img)[0]
    predictions.append(prediction4)

    # # TODO: Saving predictions
    # i = 0
    # save_path = "pictures/final_tests/predictions/"
    # print("Saving the 4 outputs as images")
    # for pred in predictions:
    #     utils.save_np_img(pred, save_path, str(i) + ".png")  # TODO: this function can't save FLOAT images
    #     i += 1


    # # Taking the BICUBIC enlargment     TODO: figure out without taking the file from path again
    # bic1 = Image.open(data_path + '11.jpg').thumbnail((img_width, img_height), Image.BICUBIC)
    # bic2 = Image.open(data_path + '12.jpg').thumbnail((img_width, img_height), Image.BICUBIC)

    # TODO: Figure out how to show images to scale
    plt.figure(figsize=(12, 12))
    plt.suptitle("Results")

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
    plt.title("WIP (ignore): 512x512")
    plt.imshow(x_test1, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(4, 4, 6)
    plt.title("WIP (ignore): 512x512")
    plt.imshow(x_test2, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(4, 4, 7)
    plt.title("WIP (ignore): 512x512")
    plt.imshow(x_test3, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)
    plt.subplot(4, 4, 8)
    plt.title("WIP (ignore): 512x512")
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

    # See: https://stackoverflow.com/a/30946248/9768291
    # plt.imsave('test.png', data, cmap = plt.cm.gray)  TODO: https://matplotlib.org/api/image_api.html#matplotlib.image.imsave
    # plt.savefig('results.png')                        TODO: https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.savefig

    plt.show()
