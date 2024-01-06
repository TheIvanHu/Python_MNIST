import numpy as np


def sigmoid(z):
    return 1/(1 +  np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def softmax(x):
    exp_values = np.exp(x - np.max(x))
    return exp_values / np.sum(exp_values)

def sigmoid_derivative(x):
    return x * (1 - x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax_derivative(z):
    s = softmax(z)
    return s * (1 - s)

def cross_entropy_loss(y, y_hat):
    epsilon = 1e-15 
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    loss = -np.sum(y * np.log(y_hat + epsilon))
    return loss

