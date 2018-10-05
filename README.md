# SR-ResCNN
A Keras implementation of a Super-Resolution Residual CNN.

The goal is to obtain a network that successfully increases the resolution of an image by filling the gaps in a manner that outperforms the generic "bicubic" method.

The model's architecture used in this project can be visualized [here](https://github.com/payne911/SR-ResCNN-Keras-/blob/master/CNN_graph.png).

# Warning
This is still a Work In Progress.

Hardware and time limitations prevent me from perfecting this project too much, but it has already reached a decent state. :)

# Demonstration
Here is a simple image of the current progress made (trained with 14 images that were data-augmented):

![comparing](https://raw.githubusercontent.com/payne911/SR-ResCNN-Keras-/master/pictures/results4.png)

The images used were, of course, never revealed to the network during training.

I am currently collecting samples coming from the more recent model that was trained using the DIV2K dataset.

### [DIV2K dataset](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf)
You can download it [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (7.1GB).

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
You will need a virtual environment that contains the following packages if you want to mess with the code:

```
tensorflow
keras
numpy
matplotlib
graphviz
scikit-image (skimage)
pillow (PIL)
hdf5 (h5py?)
```

For simple predictions through the command line, though, you will only need:

```
tensorflow
keras
numpy
matplotlib
scikit-image (skimage)
hdf5 (h5py?)
```

### Pre-trained models
I have three pre-trained models.

`my_model.h5`: An earlier one that used 14 images (that were data augmented) and the `Adam` optimizer.

`my_full_model.h5`: Another one that used a much bigger part of the real dataset I had set aside while developing the code. That one used the `Adadelta` optimizer.

Those two are actually restricted to having input of dimension ``128x128x3``.

After discussing with a friend, I realized that I could actually train this model to allow varying input sizes for predictions... hence the third model.

``unrestricted_model.h5``: This model uses the `Adadelta` optimizer and was trained with smaller sized inputs to permit bigger batch sizes (shorter train times, but more epochs required to achieve same accuracy).

They are all in the `save` folder. All these files include the weights AND the optimizer's state (so that you can train with that too).

I plan on providing the architecture as a JSON and the weights as individual files so that those can be used as "ready for integration" (I believe Android requires those files, though I still need to look that up).

I'll probably eventually provide an Android application as a Proof of Concept.

### Getting the project
Use the following line to import the project on your computer.

```
git clone https://github.com/payne911/SR-ResCNN-Keras-.git
```

You can then open it with your preferred IDE and start messing around. Minor tweaks will most probably be done through the ``constants.py`` file which is more or less the "Controler" of the flow.

### Using the model to increase the resolution of an image

As it is, I have not implemented the option to input a full-sized image. Only portions of 128x128 pixels will be shown to you as samples.

Put the image inside the ``input`` folder, and run the following command (after activating your virtual environment):

```
python predict.py your_image.png
```

This should return you a certain amount of crops from the image. Other options are available:

```
python predict.py your_image.png -a=3
```

Will show you only 3 samples from your image, for example. For a full list of the commands available to you:

```
python predict.py -h
```

### Flow of the code
To run the code through a command line, activate your virtual environment and type (though as it is it isn't really recommended since you most probably do not have a dataset set up at proper place for the default variables to work out fine... you'll have to actually modify the ``constants.py`` file):

```
python train.py
```

Based on the variables set in the ``constants.py`` file, the flow of the program will be different.

#### constants.py
For now, I'll just be lazy and copy paste its content so that you can have a rough idea of the possibilities:

```python
############################
##        PREPARE         ##
############################
y_data_path = 'dataset/DIV2K/DIV2K/to_merge/3/'  # Path from where the "y_train" data will be loaded (512x512x3 images)
hr_img_path = 'dataset/DIV2K/DIV2K/DIV2K_train_HR/'  # Path where the "y_train" data will be extracted from (if `prepare_img` is set to True)
crops_p_img = 10     # Number of samples/crops taken per HR image (to get the target output size)
augment_img = True   # Augment data with flips (each image will generate 3 more images)
prepare_img = False  # True => generate cropped images from HR (uses the paths set just above)
# # Deprecated: (used for mini tests)
# y_data_path = 'pictures/HR/512/'
# hr_img_path = 'pictures/HR/'


############################
##       SAVE/LOAD        ##
############################
load_model = True    # Should we load a saved model from memory ?
save_dir   = 'save'  # folder name where the model will be saved
model_name = 'unrestricted_model.h5'  # Name of the model that is to be loaded/saved
# TODO:
model_json = 'model_architecture.json'
weights    = 'model_weights.h5'


def get_model_save_path():
    return save_dir + '/' + model_name


############################
##         MODEL          ##
############################
img_width  = 216     # size of the output size the network will be trained for
img_height = 216     # this size divided by the scale_fact is the input size of the network
img_depth  = 3    # number of channels (RGB)
scale_fact = 4    # resolution multiplication factor
res_blocks = 3    # a power of 2, minus 1


############################
##        TRAINING        ##
############################
epochs     = 6  # 6 works well
batch_size = 7  # adjust based on your GPU memory (maximize memory usage)
verbosity  = 2  # message feedback from Keras (0, 1 or 2): higher means more verbose


############################
##       EVALUATION       ##
############################
tests_path   = 'input/'  # path to the folder containing the HR images to test with
input_width  = 128       # width size of the input used for prediction
input_height = 128       # height size of the input used for prediction
```

## Built With
* [PyCharm](https://www.jetbrains.com/pycharm/) - The IDE used
* [Keras](https://keras.io/) - The wrapper on TensorFlow framework
* [Python 3.6](https://www.python.org/downloads/release/python-360/)

## Author
* **Jérémi Grenier-Berthiaume**

See also the list of [contributors](https://github.com/payne911/SR-ResCNN-Keras-/graphs/contributors) who participated in this project.

## License
This project doesn't yet have a license applied. Reproduce at your own risk.

## Acknowledgments
My thanks go to:

* [sds](https://stackoverflow.com/users/7350191/sds): a kind StackOverflow user that has helped understand a few things when I was just starting with this project.
* [this repo](https://github.com/thstkdgus35/EDSR-PyTorch) which provided me with some insights on the actual implementation discussed in [this paper](https://arxiv.org/pdf/1707.02921.pdf).

### TODO
Notes to self.

```
* Better integrate TensorBoard (proper feature/image visualization)
* Use x_test to generate bicubic enlargments
* Use Keras ImagePreProcessing object
* Provide the "Weights" and "Architecture JSON" for both models
* Create Android Application that uses the model as a Proof of Concept
* Make a proper README.md with this @ https://gist.github.com/PurpleBooth/109311bb0361f32d87a2
```