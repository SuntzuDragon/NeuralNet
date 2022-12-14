"""net.py: Contains a class representing an artificial neural network with a single hidden layer.
           Created as the final project of MATH5810: Intro to Numerical Analysis Fall 2022
           Special thanks to Jonathan Weisberg's blog posts on neural networks.
           They were instrumental in our understanding of neural network implementation in python."""

__author__ = "Jacob Knowlton and Ian McBride"

import gzip
import time
import numpy as np
import matplotlib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm

matplotlib.use('TkAgg')


def load_images(path):
    """Loads in the images from the MNIST database source file."""
    f = gzip.open(path, 'r')

    f.read(4)
    num_images = int.from_bytes(f.read(4), 'big')
    f.read(8)
    buf = f.read(28 * 28 * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    return data.reshape((num_images, 28 * 28))


def load_labels(path):
    """Loads in the labels from the MNIST database source file."""
    f = gzip.open(path, 'r')
    f.read(4)
    num_labels = int.from_bytes(f.read(4), 'big')
    buf = f.read(num_labels)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def show_image(train_data, train_labels, index):
    """Generates a plot that shows what an image from the MNIST database looks like."""
    plt.title(train_labels[index])
    plt.imshow(train_data[index, :].reshape(28, 28), cmap=cm.binary)
    plt.axis('off')
    plt.show()


def plot_loss(net, title):
    """Plots the loss over time from a trained neural net. These values are saved in the neural net."""
    x = range(net.loss_vals.shape[0])
    y = net.loss_vals
    plt.suptitle(title)
    plt.title(f"Total training time: {net.training_time:0.4f} seconds", fontsize=8)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.plot(x, y)
    plt.show()


class NeuralNetwork:
    """Class representing an artificial neural network with one hidden layer.
    Implemented to allow either standard gradient descent or stochastic gradient descent optimization methods."""

    def __init__(self, num_hidden, train_data, train_labels, test_data, test_labels, learning_rate,
                 learn_method='gradient_descent'):
        # Set instance variables
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.learning_rate = learning_rate
        self.learn_method = learn_method
        self.training_time = 0.0
        self.loss_vals = None
        # Transform training data
        self.train_data = self.train_data.T / 255
        self.test_data = self.test_data.T / 255
        train_n = self.train_labels.shape[0]
        test_n = self.test_labels.shape[0]
        self.train_labels = self.train_labels.reshape(1, train_n)
        self.test_labels = self.test_labels.reshape(1, test_n)
        train_new = np.eye(10)[self.train_labels.astype('int32')]
        test_new = np.eye(10)[self.test_labels.astype('int32')]
        self.train_labels = train_new.T.reshape(10, train_n)
        self.test_labels = test_new.T.reshape(10, test_n)
        # Setup weights and biases
        self.w1 = np.random.randn(num_hidden, train_data.shape[1]) * np.sqrt(1.0 / train_data.shape[1])
        self.b1 = np.zeros((num_hidden, 1))
        self.w2 = np.random.randn(10, num_hidden) * np.sqrt(1.0 / train_data.shape[1])
        self.b2 = np.zeros((10, 1))
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)  # Prevents overflows
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    @staticmethod
    def __compute_loss(y, y_hat):
        """Computes the total loss of the network using cross-entropy loss."""
        loss_sum = np.sum(np.multiply(y, np.log(y_hat)))
        return -(1 / y.shape[1]) * loss_sum

    def __feed_forward(self, train_data):
        """Performs the forward pass of the neural network, calculating the output."""
        self.z1 = np.matmul(self.w1, train_data) + self.b1
        self.a1 = NeuralNetwork.sigmoid(self.z1)
        self.z2 = np.matmul(self.w2, self.a1) + self.b2
        self.a2 = NeuralNetwork.softmax(self.z2)

    def __back_prop(self, train_data, train_labels):
        """Calculates the gradient vector of the loss function, then updates the parameters using the negative
        gradient times the learning rate. """
        num_examples = train_data.shape[1]
        d_z2 = self.a2 - train_labels
        d_w2 = (1.0 / num_examples) * np.matmul(d_z2, self.a1.T)
        d_b2 = (1.0 / num_examples) * np.sum(d_z2, axis=1, keepdims=True)

        d_a1 = np.matmul(self.w2.T, d_z2)
        d_z1 = d_a1 * NeuralNetwork.sigmoid(self.z1) * (1 - NeuralNetwork.sigmoid(self.z1))
        d_w1 = (1.0 / num_examples) * np.matmul(d_z1, train_data.T)
        d_b1 = (1.0 / num_examples) * np.sum(d_z1, axis=1, keepdims=True)
        # Update parameters
        self.w2 = self.w2 - self.learning_rate * d_w2
        self.b2 = self.b2 - self.learning_rate * d_b2
        self.w1 = self.w1 - self.learning_rate * d_w1
        self.b1 = self.b1 - self.learning_rate * d_b1

    def train(self, num_epochs, verbose=False):
        """Training method for the neural network. Performs either standard gradient descent or stochastic gradient
        descent based on the parameters of the neural network. Runs epoch number of iterations and prints out the
        cost every 25 epochs if Verbose is true. """
        train_data = self.train_data
        train_labels = self.train_labels
        self.loss_vals = np.empty(num_epochs)
        start_time = time.perf_counter()
        for i in range(num_epochs):
            # Stochastic gradient descent step
            if self.learn_method == 'sgd':
                shuffle_index = np.random.permutation(self.train_data.shape[1])
                train_data = self.train_data[:, shuffle_index][:, :1000]
                train_labels = self.train_labels[:, shuffle_index][:, :1000]
            # Change weights
            self.__feed_forward(train_data)
            self.__back_prop(train_data, train_labels)
            # Compute cost
            self.__feed_forward(train_data)
            cost = self.__compute_loss(train_labels, self.a2)
            self.loss_vals[i] = cost

            if i % 25 == 0 and verbose:
                print(f"Epoch: {i}, cost: {cost}")
        end_time = time.perf_counter()
        self.training_time = end_time - start_time

    def get_results(self):
        """Prints the results of the training step. Should only be called after training has completed. Gives the
        confusion matrix of the neural network based on the testing data, as well as the overall accuracy,
        training time, and training method used. """
        self.__feed_forward(self.test_data)

        predictions = np.argmax(self.a2, axis=0)
        labels = np.argmax(self.test_labels, axis=0)

        print(f"Training method: {self.learn_method}")
        print(f"Total training time: {self.training_time:0.4f} seconds")
        print(confusion_matrix(predictions, labels))
        print(classification_report(predictions, labels))


if __name__ == '__main__':
    # Get training data
    X_train = load_images('train-images-idx3-ubyte.gz')
    y_train = load_labels('train-labels-idx1-ubyte.gz')
    X_test = load_images('t10k-images-idx3-ubyte.gz')
    y_test = load_labels('t10k-labels-idx1-ubyte.gz')

    # Check shape of data
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    # Set seed
    np.random.seed(100)

    # Create and train neural network
    sgd_net = NeuralNetwork(64, X_train, y_train, X_test, y_test, 0.5, learn_method='sgd')
    gradient_descent_net = NeuralNetwork(64, X_train, y_train, X_test, y_test, 0.5, learn_method='gradient_descent')
    sgd_net.train(3000, verbose=True)
    gradient_descent_net.train(3000, verbose=True)
    sgd_net.get_results()
    gradient_descent_net.get_results()
    plot_loss(sgd_net, "Stochastic Gradient Descent Cost Over Time")
    plot_loss(gradient_descent_net, "Gradient Descent Cost Over Time")
