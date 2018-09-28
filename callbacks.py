from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

from constants import batch_size


def get_callbacks():
    # (ml-gpu) C:\...\SR-ResCNN-Keras-\logs>
    # tensorboard --logdir .
    tbCallBack = TensorBoard(log_dir='./logs',
                             histogram_freq=1,
                             write_graph=True,
                             write_images=True,
                             write_grads=True,
                             batch_size=batch_size)

    save_callback = ModelCheckpoint("save/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='auto',
                                    period=5)  # Interval (number of epochs) between checkpoints.

    reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss',
                                     factor=0.5,  # new_lr = lr * factor
                                     patience=3,  # number of epochs with no improvement before updating
                                     verbose=1,
                                     mode='auto',
                                     min_delta=0.0001,
                                     cooldown=0,
                                     min_lr=0)

    stop_callback = EarlyStopping(monitor='val_loss',
                                  min_delta=0.00001,  # change of less than min_delta will count as no improvement
                                  patience=10,  # number of epochs with no improvement before stopping
                                  verbose=1,
                                  mode='auto',
                                  baseline=None)

    # full list:[save_callback, stop_callback, reduce_lr_cb, tbCallBack]
    return []
