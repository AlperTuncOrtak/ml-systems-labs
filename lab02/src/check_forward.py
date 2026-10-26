from common import set_seeds, load_data
import numpy as np
from nn import init_params, forward

set_seeds()
X_train, X_test, Y_train, Y_test, y_train, y_test = load_data()

for width in [16, 64]:
    params = init_params(width)
    for name, X in [("batch of 32", X_train[:32]), ("full train set", X_train)]:
        probs, _ = forward(params, X)
        max_dev = np.abs(probs.sum(axis=1) - 1).max()
        ok = probs.shape == (len(X), 10) and max_dev < 1e-6
        print(f"W={width:>2} | {name:<14} | shape {probs.shape} | "
              f"max |row sum - 1| = {max_dev:.2e} | {'PASS' if ok else 'FAIL'}")
