from common import set_seeds, load_data
from pathlib import Path
import numpy as np
import pandas as pd
from nn import init_params, forward, cross_entropy, backward

EPS = 1e-5
TOL = 1e-6
N_CHECKS = 10
RESULTS = Path(__file__).resolve().parents[1] / "results"

set_seeds()
X_train, X_test, Y_train, Y_test, y_train, y_test = load_data()
Xb, Yb = X_train[:32], Y_train[:32]

rows = []
for width in [16, 64]:
    params = init_params(width)
    probs, cache = forward(params, Xb)
    grads = backward(params, cache, Yb)

    for _ in range(N_CHECKS):
        name = np.random.choice(["W1", "W2", "W3"])
        idx = tuple(np.random.randint(s) for s in params[name].shape)
        original = params[name][idx]

        params[name][idx] = original + EPS
        loss_plus = cross_entropy(forward(params, Xb)[0], Yb)
        params[name][idx] = original - EPS
        loss_minus = cross_entropy(forward(params, Xb)[0], Yb)
        params[name][idx] = original

        numeric = (loss_plus - loss_minus) / (2 * EPS)
        analytic = grads[name][idx]
        denom = max(abs(analytic), abs(numeric))
        rel_err = 0.0 if denom == 0 else abs(analytic - numeric) / denom
        rows.append({
            "width": width, "param": name, "index": str(idx),
            "analytic": analytic, "numeric": numeric,
            "rel_error": rel_err, "pass": rel_err < TOL,
        })

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "gradient_check.csv", index=False)
pd.set_option("display.float_format", "{:.3e}".format)
print(df.to_string(index=False))
for width, g in df.groupby("width"):
    n_pass = int(g["pass"].sum())
    print(f"W={width}: {n_pass}/{N_CHECKS} passed "
          f"-> {'PASS' if n_pass >= 9 else 'FAIL'} (need >= 9)")
