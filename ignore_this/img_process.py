import itertools

from keras.layers import *
from keras.models import Model

from keras.optimizers import Adadelta
from callbacks import get_callbacks

from keras.preprocessing.image import ImageDataGenerator
from constants import img_width
from constants import img_height
from constants import img_depth
from constants import scale_fact
from constants import res_blocks
from utils import float_im

from test import run_tests


# adapted from: https://stackoverflow.com/a/52463034/9768291
def random_crop(img):
    crop_h, crop_w = img_width, img_height
    print(img.shape[0], img.shape[1])

    if (img.shape[0] >= crop_h) and (img.shape[1] >= crop_w):
        #img = rgb2ycbcr(img)  # TODO: switch from RGB channels to CbCrY
        # Cropping a random part of the image
        rand_h = np.random.randint(0, img.shape[0]-crop_h)
        rand_w = np.random.randint(0, img.shape[1]-crop_w)
        print("crops are:", rand_h, rand_w)
        tmp_img = img[rand_h:rand_h+crop_h, rand_w:rand_w+crop_w]

        new_img = float_im(tmp_img)  # From [0,255] to [0.,1.]

        # Augmenting the image  TODO: look into integrating "imgaug" library
        # TODO: I removed "augment_img" from here: ultimately remove from constants?
    else:
        return img

    return new_img


def setFakeModel():
    filters = 256
    kernel_size = 3
    strides = 1

    # TODO: To visualize internal layers (https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer/41712013#41712013)

    # Head module
    input = Input(shape=(img_height // scale_fact, img_width // scale_fact, img_depth))
    conv0 = Conv2D(filters, kernel_size, strides=strides, padding='same')(input)

    # Body module
    res = Conv2D(filters, kernel_size, strides=strides, padding='same')(conv0)
    act = ReLU()(res)
    res = Conv2D(filters, kernel_size, strides=strides, padding='same')(act)
    res_rec = Add()([conv0, res])

    for i in range(res_blocks):
        res1 = Conv2D(filters, kernel_size, strides=strides, padding='same')(res_rec)
        act = ReLU()(res1)
        res2 = Conv2D(filters, kernel_size, strides=strides, padding='same')(act)
        res_rec = Add()([res_rec, res2])

    conv = Conv2D(filters, kernel_size, strides=strides, padding='same')(res_rec)
    add = Add()([conv0, conv])

    # Tail module
    conv = Conv2D(filters, kernel_size, strides=strides, padding='same')(add)
    act = ReLU()(conv)
    up = UpSampling2D(size=scale_fact if scale_fact != 4 else 2)(act)  # TODO: try "Conv2DTranspose"
    # mul = Multiply([np.zeros((img_width,img_height,img_depth)).fill(0.1), up])(up)

    # When it's a 4X factor, we want the upscale split in two procedures
    if (scale_fact == 4):
        conv = Conv2D(filters, kernel_size, strides=strides, padding='same')(up)
        act = ReLU()(conv)
        up = UpSampling2D(size=2)(act)  # TODO: try "Conv2DTranspose"

    output = Conv2D(filters=3,
                    kernel_size=1,
                    strides=1,
                    padding='same')(up)

    train(Model(inputs=input, outputs=output))


# See: https://keras.io/preprocessing/image/
def train(model):
    print("Model done")

    # x_train = []
    # y_train = []

    # # preprocessing_function :
    # function that will be implied on each input. The function will run after the image is
    # resized and augmented. The function should take one argument: one image (Numpy tensor
    # with rank 3), and should output a Numpy tensor with the same shape.
    # we create two instances with the same arguments
    data_gen_args = dict(preprocessing_function=random_crop,
                         # rescale=1. / 255,
                         # featurewise_center=True,
                         # featurewise_std_normalization=True,
                         horizontal_flip=True,
                         vertical_flip=True,
                         validation_split=0.1)
    x_image_gen = ImageDataGenerator(**data_gen_args)
    y_image_gen = ImageDataGenerator(**data_gen_args)

    print("Before Img Gen FIT")
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    # compute quantities required for featurewise normalization (std, center)
    # x_image_gen.fit(x_train, augment=True, seed=seed)  # TODO: x_train NEED to be 4 dimensional
    # y_image_gen.fit(y_train, augment=True, seed=seed)

    x_gen = x_image_gen.flow_from_directory('pictures/keras_test',
                                            target_size=(img_width//scale_fact, img_width//scale_fact),
                                            batch_size=1,
                                            class_mode=None,  # TODO: could be "input"
                                            save_to_dir="pictures/keras_test/training/training",
                                            # save_prefix="t0_",
                                            subset="training",
                                            interpolation="lanczos",
                                            seed=seed)

    y_gen = y_image_gen.flow_from_directory('pictures/keras_test',
                                            target_size=(img_width, img_width),
                                            batch_size=1,
                                            class_mode=None,  # TODO: was None
                                            save_to_dir="pictures/keras_test/training/validation",
                                            # save_prefix="t0_",
                                            subset="training",
                                            interpolation="lanczos",
                                            seed=seed)

    print("Before Zip")
    # combine generators into one which yields x and y together
    train_generator = itertools.zip_longest(x_gen, y_gen)

    optimizer = Adadelta(lr=1.0,
                         rho=0.95,
                         epsilon=None,
                         decay=0.0)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error')

    print("Before fit_generator")
    model.fit_generator(train_generator,
                        verbose=2,
                        steps_per_epoch=12,  # equal to (nbr samples of your dataset) // (batch size)
                        epochs=6,
                        callbacks=get_callbacks())

    run_tests(model)


if __name__ == '__main__':
    setFakeModel()
