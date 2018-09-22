import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image
# from constants import img_width
# from constants import img_height
# from constants import data_path


def predict(model, X_test, Y_test):

    # print(trainingTestData.shape)  # (4, 128, 128, 3)

    # # Trying to make predictions on a bunch of images
    # predictions = model.predict(single_image)

    print("Starting predictions")

    # Extracting the benchmark images (HR)
    init1 = Y_test[0]
    init2 = Y_test[1]

    # "model.predict" works in batches, so extracting a single prediction (for memory reasons):
    img1 = X_test[0]
    img2 = X_test[1]

    input_img = (np.expand_dims(img1, 0))       # Add the image to a batch where it's the only member
    prediction1 = model.predict(input_img)[0]   # returns a list of lists, one for each image in the batch of data
    input_img = (np.expand_dims(img2, 0))
    prediction2 = model.predict(input_img)[0]

    # # Taking the BICUBIC enlargment     TODO: figure out without taking the file from path again
    # bic1 = Image.open(data_path + '11.jpg').thumbnail((img_width, img_height), Image.BICUBIC)
    # bic2 = Image.open(data_path + '12.jpg').thumbnail((img_width, img_height), Image.BICUBIC)

    # TODO: Figure out how to show images to scale
    plt.figure(figsize=(10, 10))
    plt.suptitle("Results")

    # input image
    plt.subplot(4, 2, 1)
    plt.title("Input: 128x128")
    plt.imshow(img1, cmap=plt.cm.binary)
    plt.subplot(4, 2, 2)
    plt.title("Input: 128x128")
    plt.imshow(img2, cmap=plt.cm.binary)

    # bicubic enlargment  TODO: remove duplicate (input img) placeholder
    plt.subplot(4, 2, 3)
    plt.title("Bicubic (WIP): 512x512")
    plt.imshow(img1, cmap=plt.cm.binary)
    plt.subplot(4, 2, 4)
    plt.title("Bicubic: 512x512")
    plt.imshow(img2, cmap=plt.cm.binary)

    # predicted image (x4 through network)
    plt.subplot(4, 2, 5)
    plt.title("Output: 512x512")
    plt.imshow(prediction1, cmap=plt.cm.binary)
    plt.subplot(4, 2, 6)
    plt.title("Output: 512x512")
    plt.imshow(prediction2, cmap=plt.cm.binary)

    # initial image (HR)
    plt.subplot(4, 2, 7)
    plt.title("HR version: 512x512")
    plt.imshow(init1, cmap=plt.cm.binary)
    plt.subplot(4, 2, 8)
    plt.title("HR version: 512x512")
    plt.imshow(init2, cmap=plt.cm.binary)

    plt.show()
