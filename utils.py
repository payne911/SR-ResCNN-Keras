import numpy as np

import matplotlib.image as mpimg
import skimage
from skimage import transform

from constants import data_path
from constants import img_width
from constants import img_height

from model import setUpModel


def setUpImages():

    train = []
    test = []

    sample_amnt = 11
    max_amnt = 13

    # Extracting images (512x512)
    for i in range(sample_amnt):
        train.append(mpimg.imread(data_path + str(i) + '.jpg'))

    for i in range(max_amnt-sample_amnt):
        test.append(mpimg.imread(data_path + str(i+sample_amnt) + '.jpg'))

    # Augmenting data
    trainData = dataAugmentation(train)
    testData  = dataAugmentation(test)

    setUpData(trainData, testData)


def setUpData(trainData, testData):

    # TODO: substract mean of all images to all images

    # Separating the training data
    Y_train = trainData[:len(trainData)//2]    # First half is the unaltered data
    X_train = trainData[len(trainData)//2:]    # Second half is the deteriorated data

    # Separating the testing data
    Y_test = testData[:len(testData)//2]  # First half is the unaltered data
    X_test = testData[len(testData)//2:]  # Second half is the deteriorated data

    # Adjusting shapes for Keras input  # TODO: make into a function ?
    X_train = np.array([x for x in X_train])
    Y_train = np.array([x for x in Y_train])
    Y_test = np.array([x for x in Y_test])
    X_test = np.array([x for x in X_test])

    # # Sanity check: display four images (2x HR/LR)
    # plt.figure(figsize=(10, 10))
    # for i in range(2):
    #     plt.subplot(2, 2, i + 1)
    #     plt.imshow(Y_train[i], cmap=plt.cm.binary)
    # for i in range(2):
    #     plt.subplot(2, 2, i + 1 + 2)
    #     plt.imshow(X_train[i], cmap=plt.cm.binary)
    # plt.show()

    setUpModel(X_train, Y_train, X_test, Y_test)


def dataAugmentation(dataToAugment):
    print("Starting to augment data")
    arrayToFill = []

    # faster computation with values between 0 and 1 ?
    dataToAugment = np.divide(dataToAugment, 255.)

    # TODO: switch from RGB channels to CbCrY
    # # TODO: Try GrayScale
    # trainingData = np.array(
    #     [(cv2.cvtColor(np.uint8(x * 255), cv2.COLOR_BGR2GRAY) / 255).reshape(350, 350, 1) for x in trainingData])
    # validateData = np.array(
    #     [(cv2.cvtColor(np.uint8(x * 255), cv2.COLOR_BGR2GRAY) / 255).reshape(1400, 1400, 1) for x in validateData])

    # adding the normal images   (8)
    for i in range(len(dataToAugment)):
        arrayToFill.append(dataToAugment[i])
    # vertical axis flip         (-> 16)
    for i in range(len(arrayToFill)):
        arrayToFill.append(np.fliplr(arrayToFill[i]))
    # horizontal axis flip       (-> 32)
    for i in range(len(arrayToFill)):
        arrayToFill.append(np.flipud(arrayToFill[i]))

    # downsizing by scale of 4   (-> 64 images of 128x128x3)
    for i in range(len(arrayToFill)):
        arrayToFill.append(skimage.transform.resize(
            arrayToFill[i],
            (img_width/4, img_height/4),
            mode='reflect',
            anti_aliasing=True))

    # # Sanity check: display the images
    # plt.figure(figsize=(10, 10))
    # for i in range(64):
    #     plt.subplot(8, 8, i + 1)
    #     plt.imshow(arrayToFill[i], cmap=plt.cm.binary)
    # plt.show()

    return np.array(arrayToFill)









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
