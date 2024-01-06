import csv
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data, show_image 
from loss import *
from neural_net import NeuralNetwork


def main():
    ALPHA = 0.001
    LAYERS = [(784, 64), (64, 64), (64, 10)]
    EPOCHS = 1000

    data_path = 'data.csv'
    data = load_data(data_path)

    data = np.array(data)
    X, y = data[:,1:].astype(float)/255.0, data[:,0].astype(float)
    print ('The shape of X is: ' + str(X.shape))
    print ('The shape of y is: ' + str(y.shape))

    show_image(X, y, 1)

    n = NeuralNetwork(LAYERS, ALPHA, EPOCHS)

    test_len = int(len(X)*0.1)
    train_X, train_y= X[test_len:], y[test_len:]
    test_X, test_y = X[:test_len], y[:test_len]

    n.train(train_X, train_y, test_X, test_y)
    n.test(test_X, test_y)

if __name__ == "__main__":
    main()


