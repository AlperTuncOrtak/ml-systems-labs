from common import set_seeds, load_data
import numpy as np
from nn import init_params, forward, cross_entropy

set_seeds()
X_train, X_test, Y_train, Y_test, y_train, y_test = load_data()

print(f"Expected untrained loss: ln(10) = {np.log(10):.4f}")
for width in [16, 64]:
    params = init_params(width)
    for name, X, Y in [("train", X_train, Y_train), ("test", X_test, Y_test)]:
        probs, _ = forward(params, X)
        loss = cross_entropy(probs, Y)
        print(f"W={width:>2} | {name:<5} | loss = {loss:.4f} | "
              f"diff from ln(10) = {abs(loss - np.log(10)):.2e} | "
              f"prob range [{probs.min():.4f}, {probs.max():.4f}]")
