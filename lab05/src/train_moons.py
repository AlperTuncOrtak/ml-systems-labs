import os
os.environ["OMP_NUM_THREADS"] = "1"
import random, statistics, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from engine import Value, vsum

random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.set_num_threads(1)
RESULTS = Path(__file__).resolve().parents[1] / "results"
EPOCHS, LR, HIDDEN, EPS, RUNS = 30, 0.5, 8, 1e-7, 5

X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

rng = np.random.default_rng(42)          # one shared initialisation for both
INIT = {"W1": rng.normal(0, 1, (2, HIDDEN)), "b1": np.zeros(HIDDEN),
        "W2": rng.normal(0, 0.5, (HIDDEN, 1)), "b2": np.zeros(1)}


# ---------- our engine ----------
def engine_forward(P, x):
    h = [(P["W1"][0][i] * x[0] + P["W1"][1][i] * x[1] + P["b1"][i]).relu()
         for i in range(HIDDEN)]
    return (vsum(P["W2"][i][0] * h[i] for i in range(HIDDEN)) + P["b2"][0]).sigmoid()


def engine_train():
    P = {k: [[Value(x) for x in row] for row in v] if v.ndim == 2 else [Value(x) for x in v]
         for k, v in INIT.items()}
    params = [p for v in P.values() for p in (sum(v, []) if isinstance(v[0], list) else v)]
    for _ in range(EPOCHS):
        losses = []
        for xi, yi in zip(X_tr, y_tr):
            p = engine_forward(P, xi)
            losses.append(-(yi * (p + EPS).log() + (1 - yi) * (1 - p + EPS).log()))
        loss = vsum(losses) * (1.0 / len(X_tr))
        loss.backward()
        for prm in params:
            prm.data -= LR * prm.grad
    acc = np.mean([(engine_forward(P, xi).data > 0.5) == yi for xi, yi in zip(X_te, y_te)])
    return loss.data, acc


# ---------- PyTorch, same model and algorithm ----------
Xt, yt = torch.tensor(X_tr, dtype=torch.float64), torch.tensor(y_tr, dtype=torch.float64)
Xv = torch.tensor(X_te, dtype=torch.float64)


def torch_train():
    P = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in INIT.items()}
    for _ in range(EPOCHS):
        p = torch.sigmoid(torch.relu(Xt @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]).squeeze(1)
        loss = -(yt * torch.log(p + EPS) + (1 - yt) * torch.log(1 - p + EPS)).mean()
        for v in P.values():
            v.grad = None
        loss.backward()
        with torch.no_grad():
            for v in P.values():
                v -= LR * v.grad
    with torch.no_grad():
        pv = torch.sigmoid(torch.relu(Xv @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]).squeeze(1)
    acc = float(((pv > 0.5).numpy() == y_te).mean())
    return loss.item(), acc


def timed(fn):
    fn()                                  # warm-up
    times, result = [], None
    for _ in range(RUNS):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
        print(f"  run {len(times)}: {times[-1]:.4f} s")
    return statistics.median(times), result


rows = []
for name, fn in [("our engine", engine_train), ("PyTorch", torch_train)]:
    print(name)
    med, (loss, acc) = timed(fn)
    rows.append({"implementation": name, "median_train_time_s": round(med, 4),
                 "final_train_loss": round(loss, 6), "test_accuracy": round(acc, 4)})
df = pd.DataFrame(rows)
df["slowdown_vs_pytorch"] = (df["median_train_time_s"] / df["median_train_time_s"].iloc[1]).round(0)
df.to_csv(RESULTS / "moons_speed.csv", index=False)
print(f"\n{EPOCHS} epochs, full batch, lr {LR}, {len(X_tr)} train / {len(X_te)} test samples")
print(df.to_string(index=False))
