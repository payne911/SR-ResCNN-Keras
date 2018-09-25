import tensorflow as tf
from callbacks import get_callbacks

from tests import test
from constants import verbosity
from constants import epochs
from constants import batch_size


# TODO: eventually look into https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
def train(model, x_train, y_train):

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='mean_squared_error')  # TODO: Customize loss function?
    # TODO: add custom metrics (https://keras.io/metrics/)

    # # To verify if GPU is recognized
    # tf_device = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # tf_device.list_devices()

    model.fit(x_train,
              y_train,
              epochs=epochs,  # TODO is it multi-fold testing?
              verbose=verbosity,
              shuffle=False,
              validation_split=0.1,
              batch_size=batch_size,
              callbacks=get_callbacks())
    # TODO: add save/load (https://keras.io/callbacks/#modelcheckpoint)

    test(model)
