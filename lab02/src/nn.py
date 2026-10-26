import numpy as np


def init_params(width, n_in=64, n_out=10, std=0.01):
    """Weights ~ N(0, std), biases = 0."""
    return {
        "W1": np.random.normal(0, std, (n_in, width)),
        "b1": np.zeros(width),
        "W2": np.random.normal(0, std, (width, width)),
        "b2": np.zeros(width),
        "W3": np.random.normal(0, std, (width, n_out)),
        "b3": np.zeros(n_out),
    }


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    # subtract row max for numerical stability (does not change the result)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)
    z3 = a2 @ params["W3"] + params["b3"]
    probs = softmax(z3)
    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "probs": probs}
    return probs, cache
