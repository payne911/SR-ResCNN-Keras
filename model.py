from keras.layers import *
from keras.models import Model
from keras.utils import plot_model

from constants import img_width
from constants import img_height
from constants import img_depth
from constants import res_blocks
from constants import scale_fact

from train import train


def setUpModel(X_train, Y_train, validateTestData, trainingTestData):

    # # exemple de merge de deux networks: merge = concatenate([network1, network2])
    # # exemple de deux inputs pour un seul model: model = Model(inputs=[visible1, visible2], outputs=output)

    filters = 256
    kernel_size = 3
    strides = 1

    inputLayer = Input(shape=(img_height//scale_fact, img_width//scale_fact, img_depth))
    conv1 = Conv2D(filters, kernel_size, strides=strides, padding='same')(inputLayer)

    res = Conv2D(filters, kernel_size, strides=strides, padding='same')(conv1)
    act = ReLU()(res)
    res = Conv2D(filters, kernel_size, strides=strides, padding='same')(act)
    res_rec = Add()([conv1, res])

    for i in range(res_blocks):
        res1 = Conv2D(filters, kernel_size, strides=strides, padding='same')(res_rec)
        act = ReLU()(res1)
        res2 = Conv2D(filters, kernel_size, strides=strides, padding='same')(act)
        res_rec = Add()([res_rec, res2])

    conv2 = Conv2D(filters, kernel_size, strides=strides, padding='same')(res_rec)
    a = Add()([conv1, conv2])
    up = UpSampling2D(size=scale_fact)(a)
    outputLayer = Conv2D(filters=3,
                         kernel_size=1,
                         strides=1,
                         padding='same')(up)

    model = Model(inputs=inputLayer, outputs=outputLayer)

    # Sanity checks
    print(model.summary())
    plot_model(model, to_file='CNN_graph.png')

    train(model, X_train, Y_train, validateTestData, trainingTestData)
