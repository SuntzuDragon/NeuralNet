import math
import numpy as np

class SoftMax:
    def activate(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)

class ReLU:
    def activate(self, inputs):
        return np.maximum(0, inputs)


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def feed_forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases


if __name__ == '__main__':
    pass
