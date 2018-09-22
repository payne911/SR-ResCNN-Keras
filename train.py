import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt


def train(model, X_train, Y_train, validateTestData, trainingTestData):
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',  # TODO: Customize loss function?
                  metrics=['accuracy'])

    # # Verifying if GPU is recognized
    # tf_device = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # tf_device.list_devices()

    model.fit(X_train,
              Y_train,
              epochs=5,  # TODO is it multi-fold testing?
              verbose=2,
              shuffle=False,
              batch_size=32)  # 32 images -> 1 batch
    # Now use the TEST dataset to calculate performance
    test_loss, test_acc = model.evaluate(trainingTestData, validateTestData)
    print('Test accuracy:', test_acc)


    # TODO: eventually look into https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

    ###########################
    #      PREDICTIONS        #
    ###########################

    # Trying to make predictions on a single image
    predictions = model.predict(trainingTestData)

    # "model.predict" works in batches, so extracting a single prediction:
    img = trainingTestData[0]                   # Grab an image from the test dataset
    img = (np.expand_dims(img, 0))              # Add the image to a batch where it's the only member.
    predictions_single = model.predict(img)     # returns a list of lists, one for each image in the batch of data
    print(predictions_single)

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
    plt.suptitle("Check it out!")

    # for i in range(num_images):
    #     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    #     plot_image(i, predictions, test_labels, test_images)
    #     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    #     plot_value_array(i, predictions, test_labels)
    # plt.show()
