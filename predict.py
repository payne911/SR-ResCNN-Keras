import numpy as np


def predict(model, trainingTestData):

    print(trainingTestData.shape)  # (4, 128, 128, 3)

    # Trying to make predictions on a single image
    predictions = model.predict_on_batch(trainingTestData)

    print(type(predictions))
    print(predictions)
    print(len(predictions))

    print(type(predictions[0]))
    print(predictions[0])
    print(len(predictions[0]))

    print(predictions.shape)
    print(predictions[0].shape)

    # "model.predict" works in batches, so extracting a single prediction:
    img = trainingTestData[0]  # Grab an image from the test dataset
    img = (np.expand_dims(img, 0))  # Add the image to a batch where it's the only member.
    predictions_single = model.predict(img)  # returns a list of lists, one for each image in the batch of data
    print(predictions_single)
