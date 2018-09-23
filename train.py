import tensorflow as tf
from keras.callbacks import TensorBoard

from predict import predict
from constants import epochs
from constants import batch_size


# TODO: eventually look into https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
def train(model, X_train, Y_train, X_test, Y_test):

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='mean_squared_error')  # TODO: Customize loss function?
    # TODO: add custom metrics (https://keras.io/metrics/)

    # # Verifying if GPU is recognized
    # tf_device = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # tf_device.list_devices()

    # Setting up Callbacks
    tbCallBack = TensorBoard(log_dir='./logs',  # TODO: add TensorBoard (https://keras.io/callbacks/#tensorboard)
                             histogram_freq=1,
                             write_graph=True,
                             write_images=True,
                             write_grads=True,
                             batch_size=batch_size,
                             embeddings_freq=1)

    # TODO: add "Reduced Learning Rate" (https://keras.io/callbacks/#ReduceLROnPlateau)
    # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
    #                                   min_delta=0.0001, cooldown=0, min_lr=0)

    model.fit(X_train,
              Y_train,
              epochs=epochs,  # TODO is it multi-fold testing?
              verbose=2,
              shuffle=False,
              #validation_data=(X_test, Y_test),  # TODO: verify this is a good idea to integrate there?
              batch_size=batch_size,
              callbacks=[tbCallBack])
    # TODO: add save/load (https://keras.io/callbacks/#modelcheckpoint)


    # evaluate(model, X_test, Y_test)  # TODO: fix memory problem
    predict(model, X_test, Y_test)
