import os
os.environ["OMP_NUM_THREADS"] = "1"  # avoid torch/BLAS thread oversubscription (see Lab 1)

import random

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot(y, n_classes=10):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def load_data():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0  # pixel values 0..16 -> 0..1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    return X_train, X_test, one_hot(y_train), one_hot(y_test), y_train, y_test


if __name__ == "__main__":
    set_seeds()
    X_train, X_test, Y_train, Y_test, y_train, y_test = load_data()
    print("X_train", X_train.shape, "X_test", X_test.shape)
    print("Y_train", Y_train.shape, "Y_test", Y_test.shape)
    print("X range:", X_train.min(), "to", X_train.max())
    print("One-hot rows sum to 1:", np.allclose(Y_train.sum(axis=1), 1))
    print("Train class counts:", np.bincount(y_train))
    print("Test class counts: ", np.bincount(y_test))
