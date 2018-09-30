#from keras.optimizers import Adam
from keras.optimizers import Adadelta
from callbacks import get_callbacks

from tests import test
from constants import verbosity
from constants import epochs
from constants import batch_size
from constants import load_model


# TODO: eventually look into https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
def train(model, x_train, y_train):
    print("Training is starting.")

    if load_model == False:
        print("Compiling the model since it wasn't loaded from memory.")
        # optimizer = Adam(lr=0.001,
        #                  beta_1=0.9,
        #                  beta_2=0.999,
        #                  epsilon=None,
        #                  decay=0.0,
        #                  amsgrad=False)
        optimizer = Adadelta(lr=1.0,
                             rho=0.95,
                             epsilon=None,
                             decay=0.0)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error')  # TODO: MS-SSIM loss (https://stackoverflow.com/a/51667654/9768291)

    # import keras.backend as K
    # K.set_value(optimizer.lr, 0.5 * K.get_value(optimizer.lr))

    model.fit(x_train,
              y_train,
              epochs=epochs,  # TODO is it multi-fold testing?
              verbose=verbosity,
              shuffle=False,
              validation_split=0.1,
              batch_size=batch_size,
              callbacks=get_callbacks())

    test(model)
