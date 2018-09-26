# SR-ResCNN
A Keras implementation of a Super-Resolution Residual Convolutional Neural Network.

The goal is to obtain a network that successfully increases the resolution of an image by filling the gaps in a manner that outperforms the generic "bicubic" method.

# Warning
This is still a Work In Progress. There are high probabilities of this project to be buggy. :)

### TODO

Notes to self.

```
* Better integrate TensorBoard (proper feature/image visualization)
* Use DIV2K dataset for training
* Use Keras ImagePreProcessing object
* Integrate Save/Load of Model Checkpoints
* Make a proper README.md with this @ https://gist.github.com/PurpleBooth/109311bb0361f32d87a2
```

# Demonstration

Here is a simple image of the current progress made (trained with 11 images that were data-augmented):

![comparing](https://raw.githubusercontent.com/payne911/SR-ResCNN-Keras-/master/pictures/results.png)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will need a virtual environment that contains the following packages:

```
tensorflow
numpy
matplotlib
scikit-image (skimage)
keras
pillow
hdf5 (h5py?)
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