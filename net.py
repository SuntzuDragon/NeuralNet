import gzip
import numpy as np
import matplotlib
from sklearn.metrics import classification_report, confusion_matrix

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)


def compute_loss(Y, Y_hat):
    n = Y.shape[1]
    epsilon = 1e-7
    L = -(1. / n) * (np.sum(np.multiply(np.log(Y_hat + epsilon), Y)) + np.sum(
        np.multiply(np.log(1 - Y_hat + epsilon), (1 - Y))))
    return L


def compute_multiclass_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1 / m) * L_sum
    return L


def get_results(W1, W2, b1, b2):
    Z1 = np.matmul(W1, X_test) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = softmax(Z2)

    predictions = np.argmax(A2, axis=0)
    labels = np.argmax(y_test, axis=0)

    print(confusion_matrix(predictions, labels))
    print(classification_report(predictions, labels))


def run_network(X, Y):
    learning_rate = 1

    n_x = X.shape[0]
    n_h = 64
    m = X.shape[1]

    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(10, n_h)
    b2 = np.zeros((10, 1))

    for i in range(500):
        Z1 = np.matmul(W1, X) + b1
        A1 = sigmoid(Z1)
        Z2 = np.matmul(W2, A1) + b2
        A2 = softmax(Z2)

        cost = compute_multiclass_loss(Y, A2)

        dZ2 = A2 - Y
        dW2 = (1. / m) * np.matmul(dZ2, A1.T)
        db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(W2.T, dZ2)
        dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
        dW1 = (1. / m) * np.matmul(dZ1, X.T)
        db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1

        if i % 25 == 0:
            print(f"Epoch: {i}, cost: {cost}")
    get_results(W1, W2, b1, b2)


if __name__ == '__main__':
    # Get training data
    X_train = load_images('train-images-idx3-ubyte.gz')
    y_train = load_labels('train-labels-idx1-ubyte.gz')
    X_test = load_images('t10k-images-idx3-ubyte.gz')
    y_test = load_labels('t10k-labels-idx1-ubyte.gz')

    # Transform training data
    X_train = X_train.T / 255
    X_test = X_test.T / 255
    y_train_n = y_train.shape[0]
    y_test_n = y_test.shape[0]
    y_train = y_train.reshape(1, y_train_n)
    y_test = y_test.reshape(1, y_test_n)
    y_train_new = np.eye(10)[y_train.astype('int32')]
    y_test_new = np.eye(10)[y_test.astype('int32')]
    y_train = y_train_new.T.reshape(10, y_train_n)
    y_test = y_test_new.T.reshape(10, y_test_n)

    # Check shape of data
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    m = X_train.shape[1]

    # Shuffle the data
    np.random.seed(138)
    shuffle_index = np.random.permutation(m)
    X_train, y_train = X_train[:, shuffle_index], y_train[:, shuffle_index]

    # Check plot of data
    # i = 3
    # plt.imshow(X_train[:, i].reshape(28, 28), cmap=matplotlib.cm.binary)
    # plt.axis('off')
    # plt.show()
    # print(y_train[:, i])

    run_network(X_train, y_train)
