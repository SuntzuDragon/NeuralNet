import numpy as np
import gzip


def load_images(path):
    f = gzip.open(path, 'r')

    f.read(4)
    num_images = int.from_bytes(f.read(4), 'big')
    f.read(8)
    buf = f.read(28 * 28 * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    return data.reshape((num_images, 28 * 28))


def load_labels(path):
    f = gzip.open(path, 'r')
    f.read(4)
    num_labels = int.from_bytes(f.read(4), 'big')
    buf = f.read(num_labels)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


class NeuralNet:
    def __init__(self):
        self.w1 = np.random.rand(10, 784) - 0.5
        self.b1 = np.random.rand(10, 1) - 0.5
        self.w2 = np.random.rand(10, 10) - 0.5
        self.b2 = np.random.rand(10, 1) - 0.5

        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    @staticmethod
    def relu(inputs):
        return np.maximum(0, inputs)

    @staticmethod
    def relu_prime(inputs):
        return inputs > 0

    @staticmethod
    def softmax(inputs):
        return np.divide(np.exp(inputs), np.sum(np.exp(inputs)))

    def get_predictions(self):
        return np.argmax(self.a2, 0)

    @staticmethod
    def get_accuracy(predictions, labels):
        print(predictions, labels)
        return np.sum(predictions == labels) / labels.size

    def gradient_descent(self, data, labels, iterations, alpha):
        for i in range(iterations):
            self.forward_prop(data)
            self.back_prop(data, labels, alpha)
            if i % 50 == 0:
                print(f"Iteration: {i}\tAccuracy: {NeuralNet.get_accuracy(self.get_predictions(), labels)}")

    def forward_prop(self, data):
        self.z1 = self.w1.dot(data) + self.b1
        self.a1 = NeuralNet.relu(self.z1)
        self.z2 = self.w2.dot(self.a1) + self.b2
        self.a2 = NeuralNet.softmax(self.z2)

    def back_prop(self, data, labels, alpha):
        # Calculate gradient
        one_hot_examples = NeuralNet.one_hot(labels)
        dz2 = self.a2 - one_hot_examples
        dw2 = (1 / labels.size) * dz2.dot(self.a1.T)
        db2 = (1 / labels.size) * np.sum(dz2)
        dz1 = self.w2.T.dot(dz2) * NeuralNet.relu_prime(self.z1)
        dw1 = (1 / labels.size) * dz1.dot(data.T)
        db1 = (1 / labels.size) * np.sum(dz1)

        # Update parameters
        self.w1 = self.w1 - alpha * dw1
        self.b1 = self.b1 - alpha * db1
        self.w2 = self.w2 - alpha * dw2
        self.b2 = self.b2 - alpha * db2

    @staticmethod
    def one_hot(examples):
        one_hot_examples = np.zeros((examples.size, examples.max() + 1))
        one_hot_examples[np.arange(examples.size), examples] = 1
        return one_hot_examples.T


if __name__ == '__main__':
    training_images = load_images('train-images-idx3-ubyte.gz').T
    training_labels = load_labels('train-labels-idx1-ubyte.gz')
    testing_images = load_images('t10k-images-idx3-ubyte.gz').T
    testing_labels = load_labels('t10k-labels-idx1-ubyte.gz')

    net = NeuralNet()
    net.gradient_descent(training_images, training_labels, 500, 0.1)
