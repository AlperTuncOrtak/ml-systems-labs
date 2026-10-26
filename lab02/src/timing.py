from common import set_seeds, load_data
from pathlib import Path
import statistics
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nn import init_params, forward, backward

REPEATS = 100
RESULTS = Path(__file__).resolve().parents[1] / "results"

set_seeds()
X_train, X_test, Y_train, Y_test, y_train, y_test = load_data()


def time_ms(fn, repeats=REPEATS):
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


rows = []
for width in [16, 64]:
    params = init_params(width)
    for batch in [32, len(X_train)]:
        X, Y = X_train[:batch], Y_train[:batch]
        _, c = forward(params, X)  # cache for backward, outside timing

        # intermediate gradients, so backward matmuls can be timed alone
        dz3 = (c["probs"] - Y) / batch
        dz2 = (dz3 @ params["W3"].T) * (c["z2"] > 0)
        dz1 = (dz2 @ params["W2"].T) * (c["z1"] > 0)

        fwd = time_ms(lambda: forward(params, X))
        bwd = time_ms(lambda: backward(params, c, Y))
        fwd_mm = time_ms(lambda: (X @ params["W1"],
                                  c["a1"] @ params["W2"],
                                  c["a2"] @ params["W3"]))
        bwd_mm = time_ms(lambda: (c["a2"].T @ dz3, dz3 @ params["W3"].T,
                                  c["a1"].T @ dz2, dz2 @ params["W2"].T,
                                  X.T @ dz1))

        fwd_macs = batch * (64 * width + width * width + 10 * width)
        bwd_macs = batch * (64 * width + 2 * width * width + 20 * width)
        rows.append({
            "width": width, "batch": batch,
            "forward_ms": round(fwd, 4), "backward_ms": round(bwd, 4),
            "bwd_fwd_ratio": round(bwd / fwd, 2),
            "forward_matmul_ms": round(fwd_mm, 4),
            "backward_matmul_ms": round(bwd_mm, 4),
            "matmul_bwd_fwd_ratio": round(bwd_mm / fwd_mm, 2),
            "theoretical_MAC_ratio": round(bwd_macs / fwd_macs, 2),
            "forward_matmul_share": round(fwd_mm / fwd, 2),
            "backward_matmul_share": round(bwd_mm / bwd, 2),
            "forward_MACs": fwd_macs, "backward_MACs": bwd_macs,
        })

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "timing.csv", index=False)
print(df.drop(columns=["forward_MACs", "backward_MACs"]).T.to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, batch in zip(axes, [32, len(X_train)]):
    d = df[df["batch"] == batch].reset_index(drop=True)
    x = np.arange(len(d))
    for offset, part in [(-0.2, "forward"), (0.2, "backward")]:
        mm = d[f"{part}_matmul_ms"]
        other = d[f"{part}_ms"] - mm
        ax.bar(x + offset, mm, 0.4,
               color="C0" if part == "forward" else "C1",
               label=f"{part}: matmul")
        ax.bar(x + offset, other, 0.4, bottom=mm,
               color="C0" if part == "forward" else "C1", alpha=0.4,
               hatch="//", label=f"{part}: other ops")
    ax.set_xticks(x)
    ax.set_xticklabels([f"W={w}" for w in d["width"]])
    ax.set_title(f"Batch = {batch}")
    ax.set_ylabel("Median time (ms, 100 runs)")
    ax.grid(axis="y", alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle("Forward vs backward time, split into matrix multiplication and other ops")
fig.tight_layout()
fig.savefig(RESULTS / "timing.png", dpi=150)
print("Saved timing.png")
