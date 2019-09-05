from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from constants import batch_size, add_callbacks, hr_img_path, val_split, log_dir
from generator import ImgDataGenerator


def get_callbacks():
    if add_callbacks:
        # (ml-gpu) C:\...\SR-ResCNN-Keras-\logs>
        # tensorboard --logdir .
        tbCallBack = TensorBoard(log_dir=log_dir,
                                 histogram_freq=0,  # epoch-frequency of calculations
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
                                         factor=0.1,  # new_lr = lr * factor
                                         patience=4,  # number of epochs with no improvement before updating
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

        import fnmatch
        import os
        nb_samples = len(fnmatch.filter(os.listdir(hr_img_path), '*.png'))
        train_gen, val_gen = ImgDataGenerator(hr_img_path,
                                              validation_split=val_split,
                                              nb_samples=nb_samples).get_all_generators()

        diagnose_cb = ModelDiagnoser(train_gen,    # data_generator
                                     batch_size,   # batch_size
                                     nb_samples,   # num_samples
                                     log_dir,      # output_dir
                                     0)            # normalization_mean

        # To include the full list:
        # return [save_callback, stop_callback, reduce_lr_cb, tbCallBack, diagnose_cb]
        return [tbCallBack, diagnose_cb]
    else:
        return None


# The "Model Diagnoser" sends sample images to the Tensorboard
# see https://stackoverflow.com/a/55856716/9768291
import os

import io
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.callbacks import Callback
from keras.utils import GeneratorEnqueuer, Sequence, OrderedEnqueuer


def make_image_tensor(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Adapted from https://github.com/lanpa/tensorboard-pytorch/
    """
    if len(tensor.shape) == 3:
        height, width, channel = tensor.shape
    else:
        height, width = tensor.shape
        channel = 1
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class TensorBoardWriter:
    def __init__(self, outdir):
        assert (os.path.isdir(outdir))
        self.outdir = outdir
        self.writer = tf.summary.FileWriter(self.outdir,
                                            flush_secs=10)

    def save_image(self, tag, image, global_step=None):
        image_tensor = make_image_tensor(image)
        self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, image=image_tensor)]),
                                global_step)

    def close(self):
        """
        To be called in the end
        """
        self.writer.close()


class ModelDiagnoser(Callback):
    def __init__(self,
                 data_generator,
                 m_batch_size,
                 num_samples,
                 output_dir,
                 normalization_mean):
        self.epoch_index = 0
        self.data_generator = data_generator
        self.batch_size = m_batch_size
        self.num_samples = num_samples
        self.tensorboard_writer = TensorBoardWriter(output_dir)
        self.normalization_mean = normalization_mean
        is_sequence = isinstance(self.data_generator, Sequence)
        if is_sequence:
            self.enqueuer = OrderedEnqueuer(self.data_generator,
                                            use_multiprocessing=True,
                                            shuffle=False)
        else:
            self.enqueuer = GeneratorEnqueuer(self.data_generator,
                                              use_multiprocessing=False,  # todo: how to 'True' ?
                                              wait_time=0.01)
        self.enqueuer.start(workers=4, max_queue_size=4)

    def on_epoch_end(self, epoch, logs=None):
        output_generator = self.enqueuer.get()
        steps_done = 0
        total_steps = int(np.ceil(np.divide(self.num_samples, self.batch_size)))
        sample_index = 0
        while steps_done < total_steps:
            generator_output = next(output_generator)
            x, y = generator_output[:2]
            x = next(iter(x.values()))
            y = next(iter(y.values()))
            y_pred = self.model.predict(x)
            self.epoch_index += 1

            for i in range(0, len(y_pred)):
                n = steps_done * self.batch_size + i
                if n >= self.num_samples:
                    return

                # rearranging images for visualization
                img_x = self.__reformat_img(x, i)
                img_y = self.__reformat_img(y, i)
                img_p = self.__reformat_img(y_pred, i)

                self.tensorboard_writer.save_image("Epoch-{}/{}/x"
                                                   .format(self.epoch_index, sample_index), img_x)
                self.tensorboard_writer.save_image("Epoch-{}/{}/y"
                                                   .format(self.epoch_index, sample_index), img_y)
                self.tensorboard_writer.save_image("Epoch-{}/{}/y_pred"
                                                   .format(self.epoch_index, sample_index), img_p)
                sample_index += 1

            steps_done += 1

    def __reformat_img(self, img_np_array, img_index):
        img = np.squeeze(img_np_array[img_index, :, :, :])
        img = 255. * (img + self.normalization_mean)  # mean is the training images normalization mean
        img = img[:, :, [2, 1, 0]]  # reordering of channels
        return img

    def on_train_end(self, logs=None):
        self.enqueuer.stop()
        self.tensorboard_writer.close()
