import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
#from PIL import Image

import utils
from constants import img_width
from constants import img_height
from constants import batch_size
from constants import verbosity
from constants import get_model_save_path
from constants import tests_path


# TODO: Add an `args.parser` to be able to predict directly from command prompt
# TODO: Add a function to take images from folder and test with them directly (no ground_truth)
def run_tests(model):
    x_test, y_test = extract_tests()

    evaluate(model, x_test, y_test)
    predicts(model, x_test, y_test)


def extract_tests():
    print("Starting tests. Extracting images to feed.")
    x = []
    y = []

    for i in range(11):
        # Extracting the benchmark images (HR)
        y_test = utils.crop_center(skimage.io.imread(tests_path + str(i) + ".png"), img_width, img_height)
        y.append(y_test)
        # Extracting middle part for prediction test
        x.append(utils.single_downscale(y_test))

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


def predicts(model, x_test, y_test):
    print("Starting predictions.")

    # # Trying to make predictions on a bunch of images (works in batches)
    # predictions = model.predict(images)

    # Extracting predictions
    predictions = []
    for i in range(len(x_test)):
        input_img = (np.expand_dims(x_test[i], 0))       # Add the image to a batch where it's the only member
        predictions.append(model.predict(input_img)[0])  # returns a list of lists, one for each image in the batch

    # # Taking the BICUBIC enlargment     TODO: figure out without taking the file from path again
    # bic1 = Image.open(data_path + '11.jpg').thumbnail((img_width, img_height), Image.BICUBIC)
    # bic2 = Image.open(data_path + '12.jpg').thumbnail((img_width, img_height), Image.BICUBIC)

    # # TODO: Saving predictions
    # i = 0
    # save_path = "pictures/final_tests/predictions/"
    # print("Saving the 4 outputs as images")
    # for pred in predictions:
    #     utils.save_np_img(pred, save_path, str(i) + ".png")
    #     i += 1

    # https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.savefig
    # plt.savefig('pictures/final_tests/predictions/results.png', frameon=True) TODO: not working (white image)

    # Showing output vs expected image
    for i in range(len(predictions)):
        show_pred_output(x_test[i], predictions[i], y_test[i])

    prompt_model_save(model)


def show_pred_output(input, pred, truth):
    plt.figure(figsize=(20, 20))
    plt.suptitle("Results")

    plt.subplot(1, 3, 1)
    plt.title("Input: 128x128")
    plt.imshow(input, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    plt.subplot(1, 3, 2)
    plt.title("Output: 512x512")
    plt.imshow(pred, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    plt.subplot(1, 3, 3)
    plt.title("Target (HR): 512x512")
    plt.imshow(truth, cmap=plt.cm.binary).axes.get_xaxis().set_visible(False)

    plt.show()


def prompt_model_save(model):
    save_bool = input("Save progress from this model (y/n) ?\n")
    if save_bool == "y":
        model.save(get_model_save_path())
        print("Model saved! :)")
        # model.save_weights('save/model_weights.h5')
    del model  # deletes the existing model  # TODO: use it even if not saving?


# def predict(img_path):
#     return img_path
#
# if __name__ == '__main__':
#     predict(args)
