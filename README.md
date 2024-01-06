# Python_MNIST
Python implementation of MNIST, without ML libraries 

MNIST is a data set of handwritten digits, along with the corresponding labels as to which digit it is. In this project, a neural network, with architecture of four layers, two of which are hidden, of 784, 64, 64, 10 neurons, is trained in Python. No regularily used machine learning libraries are used, such as PyTorch, Scikit-Learn, Tensorflow, the only datascience library used is numpy in order to speed up matrix operations.
The other libraries used is csvin order to process input files, as well as matplotlib in order to show the images. 

**Files**
1. utils.py contains the scripts necessary for loading the data, as well as showing the images using plt
2. loss.py contains various different loss functions and their derivatives, including sigmoid, relu, softmax
3. neural_net.py contains the class of NeuralNetwork, which itself contains the functions for both forward propogation and back propogation, currently using the sigmoid function as the activation.
4. training.py contains the main function for training the neural network, as well as giving options for changing the parameters for alpha, the layers, and the number of epochs
