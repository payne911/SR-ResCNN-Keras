# import os.path
# from constants import save_dir
# from constants import model_json
# from constants import weights

from keras.layers import *
from keras.models import Model, load_model
from keras.utils import plot_model
from constants import get_model_save_path, img_depth, res_blocks, scale_fact
from train import generator_train


def setUpModel():
    print("Setting up the CNN.")

    filters = 256    # amount of filters outputted
    kernel_size = 3  # (3x3) is the kernel size
    strides = 1

    # Head module
    input = Input(name='input',
                  shape=(None, None, img_depth))  # None = can take inputs of different sizes
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
    act  = ReLU()(conv)
    up   = UpSampling2D(size=scale_fact if scale_fact != 4 else 2)(act)  # TODO: try "Conv2DTranspose"
    # mul = Multiply([np.zeros((img_width,img_height,img_depth)).fill(0.1), up])(up)

    # When it's a 4X factor, we want the upscale split in two procedures
    if(scale_fact == 4):
        conv = Conv2D(filters, kernel_size, strides=strides, padding='same')(up)
        act  = ReLU()(conv)
        up   = UpSampling2D(size=2)(act)  # TODO: try "Conv2DTranspose"

    output = Conv2D(name='output',
                    filters=3,
                    kernel_size=1,
                    strides=1,
                    padding='same')(up)

    model = Model(inputs=input, outputs=output)
    sanity_checks(model)

    # save_arch_and_weights(model)  # TODO: good place? necessary?

    generator_train(model)


def load_gen_model():
    print("Loading model from memory (with generator).")
    model = load_model(get_model_save_path())  # TODO: add try-catch in case of wrong variable
    sanity_checks(model)

    generator_train(model)


def sanity_checks(model):
    print(model.summary())
    plot_model(model, to_file='CNN_graph.png')


# def save_arch_and_weights(model):
    # # Save the model architecture (JSON)
    # model_path = save_dir + '/' + model_json
    # with open(model_path, 'w') as f:
    #     f.write(model.to_json())

    # from keras.models import model_from_json
    # # Model reconstruction from JSON file
    # with open('model_architecture.json', 'r') as f:
    #     model = model_from_json(f.read())

    # # Load weights into the new model
    # save_path = save_dir + '/' + weights
    # if os.path.isfile(save_path):
    #     print("Loading weights from previously saved model.")
    #     model.load_weights(save_path)
