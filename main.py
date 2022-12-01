import math
import numpy as np

class SoftMax:
    def activate(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)

class ReLU:
    def activate(self, inputs):
        return np.maximum(0, inputs)

class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        return -np.log(correct_confidences)
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def feed_forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases


if __name__ == '__main__':
    pass
