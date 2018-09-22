import numpy as np

import matplotlib.image as mpimg
import skimage
from skimage import transform

from constants import data_path
from constants import img_width
from constants import img_height

from model import setUpModel


def setUpImages():

    # Setting up paths
    path1  = data_path + 'Moi.jpg'
    path2  = data_path + 'ASaucerfulOfSecrets.jpg'
    path3  = data_path + 'AtomHeartMother.jpg'
    path4  = data_path + 'Animals.jpg'
    path5  = data_path + 'DivisionBell.jpg'          # validator
    path6  = data_path + 'lighter.jpg'
    path7  = data_path + 'Meddle.jpg'                # validator
    path8  = data_path + 'ObscuredByClouds.jpg'      # validator
    path9  = data_path + 'TheDarkSideOfTheMoon.jpg'
    path10 = data_path + 'TheWall.jpg'
    path11 = data_path + 'WishYouWereHere.jpg'

    # Extracting images (1400x1400)
    train = [mpimg.imread(path1),
             mpimg.imread(path2),
             mpimg.imread(path3),
             mpimg.imread(path4),
             mpimg.imread(path6),
             mpimg.imread(path9),
             mpimg.imread(path10),
             mpimg.imread(path11)]
    finalTest = [mpimg.imread(path5),
                 mpimg.imread(path8),
                 mpimg.imread(path7)]

    # Augmenting data
    trainData = dataAugmentation(train)
    testData  = dataAugmentation(finalTest)

    setUpData(trainData, testData)


def setUpData(trainData, testData):

    # print(type(trainData))                          # <class 'numpy.ndarray'>
    # print(len(trainData))                           # 64
    # print(type(trainData[0]))                       # <class 'numpy.ndarray'>
    # print(trainData[0].shape)                       # (1400, 1400, 3)
    # print(trainData[len(trainData)//2-1].shape)     # (1400, 1400, 3)
    # print(trainData[len(trainData)//2].shape)       # (350, 350, 3)
    # print(trainData[len(trainData)-1].shape)        # (350, 350, 3)

    # TODO: substract mean of all images to all images

    # Separating the training data
    Y_train = trainData[:len(trainData)//2]    # First half is the unaltered data
    X_train = trainData[len(trainData)//2:]    # Second half is the deteriorated data

    # Separating the testing data TODO: rename variables to conventions?
    validateTestData = testData[:len(testData)//2]  # First half is the unaltered data
    trainingTestData = testData[len(testData)//2:]  # Second half is the deteriorated data

    # Adjusting shapes for Keras input
    X_train = np.array([x for x in X_train])
    Y_train = np.array([x for x in Y_train])

    # # Sanity check: display four images (2x HR/LR)
    # plt.figure(figsize=(10, 10))
    # for i in range(2):
    #     plt.subplot(2, 2, i + 1)
    #     plt.imshow(Y_train[i], cmap=plt.cm.binary)
    # for i in range(2):
    #     plt.subplot(2, 2, i + 1 + 2)
    #     plt.imshow(X_train[i], cmap=plt.cm.binary)
    # plt.show()

    setUpModel(X_train, Y_train, validateTestData, trainingTestData)


# see: https://keras.io/preprocessing/image/
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

    # downsizing by scale of 4   (-> 64 images of 350x350x3)
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
