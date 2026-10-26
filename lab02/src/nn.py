import numpy as np


def init_params(width, n_in=64, n_out=10, std=0.1):
    """Weights ~ N(0, 0.01) i.e. variance 0.01, std 0.1. Biases = 0."""
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


def cross_entropy(probs, Y):
    """Mean cross-entropy. Y is one-hot. Clip avoids log(0)."""
    return -np.mean(np.sum(Y * np.log(np.clip(probs, 1e-12, 1.0)), axis=1))


def backward(params, cache, Y):
    """Gradients of mean cross-entropy w.r.t. all parameters (chain rule)."""
    N = Y.shape[0]
    dz3 = (cache["probs"] - Y) / N              # softmax + cross-entropy
    grads = {"W3": cache["a2"].T @ dz3, "b3": dz3.sum(axis=0)}

    da2 = dz3 @ params["W3"].T
    dz2 = da2 * (cache["z2"] > 0)               # ReLU derivative
    grads["W2"] = cache["a1"].T @ dz2
    grads["b2"] = dz2.sum(axis=0)

    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (cache["z1"] > 0)
    grads["W1"] = cache["X"].T @ dz1
    grads["b1"] = dz1.sum(axis=0)
    return grads
