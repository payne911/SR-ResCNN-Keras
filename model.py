from keras.layers import *
from keras.models import Model
from keras.utils import plot_model

from constants import img_width
from constants import img_height
from constants import img_depth
from constants import res_blocks
from constants import scale_fact

from train import train


# TODO: Adapt to https://github.com/jmiller656/EDSR-Tensorflow/blob/master/utils.py
def setUpModel(x_train, y_train):
    print("Setting up the Neural Network.")

    # # exemple de merge de deux networks: merge = concatenate([network1, network2])
    # # exemple de deux inputs pour un seul model: model = Model(inputs=[visible1, visible2], outputs=output)

    filters = 256
    kernel_size = 3
    strides = 1

    # TODO: To visualize internal layers (https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer/41712013#41712013)

    # Head module
    input = Input(shape=(img_height//scale_fact, img_width//scale_fact, img_depth))
    conv0 = Conv2D(filters, kernel_size, strides=strides, padding='same')(input)

    # Body module
    res = Conv2D(filters, kernel_size, strides=strides, padding='same')(conv0)
    act = ReLU()(res)
    res = Conv2D(filters, kernel_size, strides=strides, padding='same')(act)
    res_rec = Add()([conv0, res])

    for i in range(res_blocks):
        res1 = Conv2D(filters, kernel_size, strides=strides, padding='same')(res_rec)
        act  = ReLU()(res1)
        res2 = Conv2D(filters, kernel_size, strides=strides, padding='same')(act)
        res_rec = Add()([res_rec, res2])

    conv = Conv2D(filters, kernel_size, strides=strides, padding='same')(res_rec)
    add  = Add()([conv0, conv])

    # Tail module
    conv = Conv2D(filters, kernel_size, strides=strides, padding='same')(add)
    act = ReLU()(conv)
    up  = UpSampling2D(size=scale_fact if scale_fact != 4 else 2)(act)  # TODO: try "Conv2DTranspose"
    # mul = Multiply([np.zeros((img_width,img_height,img_depth)).fill(0.1), up])(up)

    # When it's a 4X factor, we want the upscale split in two procedures
    if(scale_fact == 4):
        conv = Conv2D(filters, kernel_size, strides=strides, padding='same')(up)
        act  = ReLU()(conv)
        up   = UpSampling2D(size=2)(act)  # TODO: try "Conv2DTranspose"

    output = Conv2D(filters=3,
                    kernel_size=1,
                    strides=1,
                    padding='same')(up)

    model = Model(inputs=input, outputs=output)

    # Sanity checks
    print(model.summary())
    plot_model(model, to_file='CNN_graph.png')

    train(model, x_train, y_train)
