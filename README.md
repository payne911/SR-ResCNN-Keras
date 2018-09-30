# SR-ResCNN
A Keras implementation of a Super-Resolution Residual Convolutional Neural Network.

The goal is to obtain a network that successfully increases the resolution of an image by filling the gaps in a manner that outperforms the generic "bicubic" method.

# Warning
This is still a Work In Progress.

# Demonstration

Here is a simple image of the current progress made (trained with 14 images that were data-augmented):

![comparing](https://raw.githubusercontent.com/payne911/SR-ResCNN-Keras-/master/pictures/results3.png)

One extra training iteration (of 6 epochs) gave the following result:

![comparing](https://raw.githubusercontent.com/payne911/SR-ResCNN-Keras-/master/pictures/results4.png)

The images used were, of course, never revealed to the network during training.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will need a virtual environment that contains the following packages:

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

### Pre-trained models

I have two pre-trained models. An earlier one that used 14 images (that were data augmented) and the `Adam` optimizer.

And another one that used the whole real data set I had set aside while developing the code. That one used the `Adadelta` optimizer.

They are both in the `save` folder. The first one is `my_model.h5` and the second one is `my_full_model.h5`. Both these files include the weights AND the optimizer's state (so that you can train with that too).

I plan on providing the architecture as a JSON and the weights as individual files so that those can be used as "ready for integration" (I believe Android requires those files, though I still need to look that up).

I'll probably eventually provide an Android application as a Proof of Concept.

### Getting the project

Use the following line to import the project on your computer.

```
git clone https://github.com/payne911/SR-ResCNN-Keras-.git
```

You can then open it with your preferred IDE and start messing around. Minor tweaks will most probably be done through the ``constants.py`` file which is more or less the "Controler" of the flow.

### Flow of the code

To run the code through a command line, activate your virtual environment and type:

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
y_data_path = 'dataset/DIV2K/DIV2K/to_merge/1/'  # Path from where the "y_train" data will be loaded (512x512x3 images)
hr_img_path = 'dataset/DIV2K/DIV2K/DIV2K_train_HR/'  # Path where the "y_train" data will be extracted from (if `prepare_img` is set to True)
crops_p_img = 7      # Number of samples/crops taken per HR image (to get the target output size)
augment_img = True   # Augment data with flips (each image will generate 3 more images)
prepare_img = False  # True => generate cropped images from HR (uses the paths set just above)
# # Deprecated: (used for mini tests)
# y_data_path = 'pictures/HR/512/'
# hr_img_path = 'pictures/HR/'


############################
##       SAVE/LOAD        ##
############################
load_model  = True  # Should we load a saved model from memory ?
save_dir    = 'save'  # folder name where the model will be saved
model_name  = 'my_full_model.h5'  # Name of the model that is to be loaded/saved
# TODO:
model_json  = 'model_architecture.json'
weights     = 'model_weights.h5'


def get_model_save_path():
    return save_dir + '/' + model_name


############################
##         MODEL          ##
############################
img_width  = 512     # size of the output size the network will be trained for
img_height = 512     # this size divided by the scale_fact is the input size of the network
img_depth  = 3    # number of channels (RGB)
scale_fact = 4    # resolution multiplication factor
res_blocks = 3    # a power of 2, minus 1


############################
##        TRAINING        ##
############################
epochs     = 6  # 6 works well
batch_size = 1  # adjust based on your GPU memory (maximize memory usage)
verbosity  = 2  # message feedback from Keras (0, 1 or 2): higher means more verbose
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

Thanks to everyone that has helped me through this project.

### TODO

Notes to self.

```
* Better integrate TensorBoard (proper feature/image visualization)
* Use Keras ImagePreProcessing object
* Provide the "Weights" and "Architecture JSON" for both models
* Add "arg.parser" to facilitate command-line control
* Create Android Application that uses the model as a Proof of Concept
* Make a proper README.md with this @ https://gist.github.com/PurpleBooth/109311bb0361f32d87a2
```