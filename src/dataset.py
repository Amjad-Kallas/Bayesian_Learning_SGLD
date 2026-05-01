import numpy as np


def load_a9a(path, n_features=123):
    """
    Load the a9a dataset in LIBSVM format.
    """

    X = []
    y = []

    
    with open(path, "r") as f:

        for line in f:

            sample = line.strip().split()

            label = int(sample[0])
            y.append(label)


            features = np.zeros(n_features)

            for item in sample[1:]:
                idx, val = item.split(":")
                idx = int(idx) - 1
                features[idx] = float(val)

            X.append(features)

    X = np.array(X)
    y = np.array(y)

    return X, y

def add_bias(X):
    """
    Add bias column for LR (as done in the paper).
    """

    N = X.shape[0]

    bias = np.ones((N, 1))

    return np.hstack((X, bias))


def train_test_split(X, y, test_ratio=0.2):
    """
    Random 80/20 split training/test as used in the paper.
    """

    N = X.shape[0]

    idx = np.random.permutation(N)

    split = int((1 - test_ratio) * N)

    train_idx = idx[:split]
    test_idx = idx[split:]

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def minibatch(X, y, batch_size):
    """
    Sample a minibatch.
    This will be used to estimate the gradient using random minibatches at each iteration
    """

    N = X.shape[0]

    idx = np.random.choice(N, batch_size, replace=True)

    return X[idx], y[idx]


