import tensorflow as tf

from predict import predict
from constants import epochs
from constants import batch_size


# TODO: eventually look into https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
def train(model, X_train, Y_train, trainingTestData, validateTestData):

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='mean_squared_error')  # TODO: Customize loss function?
    # TODO: add custom metrics : https://keras.io/metrics/

    # # Verifying if GPU is recognized
    # tf_device = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # tf_device.list_devices()

    model.fit(X_train,
              Y_train,
              epochs=epochs,  # TODO is it multi-fold testing?
              verbose=2,
              shuffle=False,
              batch_size=batch_size)

    # evaluate(model, trainingTestData, validateTestData)  # TODO: fix memory problem
    predict(model, trainingTestData)
