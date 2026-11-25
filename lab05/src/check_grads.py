import os
os.environ["OMP_NUM_THREADS"] = "1"
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from engine import Value, vsum

random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
RESULTS = Path(__file__).resolve().parents[1] / "results"
SHAPE, EPS, TOL = (3, 4), 1e-5, 1e-6

A, B, C = (np.random.normal(0, 1, SHAPE) for _ in range(3))

# 1) our engine
va, vb, vc = ([[Value(x) for x in row] for row in M] for M in (A, B, C))
out = vsum(((va[i][j] * vb[i][j] + vc[i][j]) ** 2).relu()
           for i in range(SHAPE[0]) for j in range(SHAPE[1]))
graph = out.backward()
ours = {n: np.array([[v.grad for v in row] for row in V]) for n, V in
        [("a", va), ("b", vb), ("c", vc)]}

# 2) PyTorch autograd (float64)
ta, tb, tc = (torch.tensor(M, dtype=torch.float64, requires_grad=True) for M in (A, B, C))
t_out = ((ta * tb + tc) ** 2).relu().sum()
t_out.backward()
theirs = {"a": ta.grad.numpy(), "b": tb.grad.numpy(), "c": tc.grad.numpy()}

# 3) numerical gradients (central difference)
def f(A, B, C):
    return np.sum(np.maximum(0, (A * B + C) ** 2))
numeric = {}
for name, M in [("a", A), ("b", B), ("c", C)]:
    g = np.zeros(SHAPE)
    for idx in np.ndindex(SHAPE):
        old = M[idx]
        M[idx] = old + EPS; plus = f(A, B, C)
        M[idx] = old - EPS; minus = f(A, B, C)
        M[idx] = old
        g[idx] = (plus - minus) / (2 * EPS)
    numeric[name] = g

rows = []
for n in "abc":
    d_t = np.abs(ours[n] - theirs[n]).max()
    d_n = np.abs(ours[n] - numeric[n]).max()
    rows.append({"input": n, "max_abs_diff_vs_pytorch": d_t, "max_abs_diff_vs_numeric": d_n,
                 "pass": bool(d_t < TOL and d_n < TOL)})
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "gradient_check.csv", index=False)

# accumulation check: d(a*a)/da = 2a, needs += to be correct
x = Value(3.0); y = x * x; y.backward()
acc_ok = abs(x.grad - 6.0) < 1e-12

pd.set_option("display.float_format", "{:.2e}".format)
print(f"forward value: ours {out.data:.10f} | pytorch {t_out.item():.10f}")
print(f"graph nodes: {len(graph)}")
print(df.to_string(index=False))
print(f"accumulation check d(x*x)/dx at x=3: {x.grad} (expected 6.0) -> {'PASS' if acc_ok else 'FAIL'}")
print("OVERALL:", "PASS" if df["pass"].all() and acc_ok else "FAIL")
