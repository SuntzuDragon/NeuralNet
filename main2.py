import numpy as np
import gzip
import matplotlib.pyplot as plt
import random


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


if __name__ == '__main__':
    training_images = load_images('train-images-idx3-ubyte.gz').T
    training_labels = load_labels('train-labels-idx1-ubyte.gz')
    testing_images = load_images('t10k-images-idx3-ubyte.gz').T
    testing_labels = load_labels('t10k-labels-idx1-ubyte.gz')

    print(training_images.shape)
    print(training_labels.shape)
    print(testing_images.shape)
    print(testing_labels.shape)
