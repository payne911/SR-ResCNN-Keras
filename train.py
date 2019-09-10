from keras.optimizers import Adadelta
from callbacks import get_callbacks

from test import run_tests
from constants import verbosity, epochs, batch_size,\
    load_model, hr_img_path, val_split

from generator import ImgDataGenerator


def generator_train(model):
    print("\n\nTraining is starting.")

    if load_model == False:
        print("Compiling the model since it wasn't loaded from memory.")
        optimizer = Adadelta(lr=1.0,
                             rho=0.95,
                             epsilon=None,
                             decay=0.0)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error')  # TODO: MS-SSIM loss (https://stackoverflow.com/a/51667654/9768291)

    import fnmatch
    import os
    nb_samples = len(fnmatch.filter(os.listdir(hr_img_path), '*.png'))
    print("Number of samples:", nb_samples)
    import math
    print("Steps per epoch:", math.ceil(nb_samples / batch_size))

    train_gen, val_gen = ImgDataGenerator(hr_img_path,
                                          validation_split=val_split,
                                          nb_samples=nb_samples,
                                          random_samples=False).get_all_generators()
    train_steps_per_epoch = math.ceil(nb_samples / batch_size)
    val_steps_per_epoch   = math.ceil((nb_samples - int(val_split * nb_samples)) / batch_size)

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=train_steps_per_epoch,  # number of batches coming out of generator
                                  epochs=epochs,
                                  validation_data=val_gen,
                                  validation_steps=val_steps_per_epoch,
                                  verbose=verbosity,
                                  shuffle=False,
                                  callbacks=get_callbacks())

    run_tests(model, history)
