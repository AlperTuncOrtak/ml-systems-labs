from common import RESULTS
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cost = pd.read_csv(RESULTS / "cost_summary.csv")
acc = pd.read_csv(RESULTS / "test_accuracy.csv")
df = cost.merge(acc, on="model")

# achieved throughput at batch 128: FLOPs per batch / time
df["GFLOPs_per_s_b128"] = (df["hand_FLOPs"] * 128 / (df["latency_b128_ms"] / 1000) / 1e9).round(1)
df["acc_per_100k_params"] = (df["test_acc"] / (df["params"] / 1e5)).round(3)

cols = ["model", "params", "hand_FLOPs", "test_acc", "latency_b1_ms",
        "latency_b128_ms", "per_sample_b128_ms", "GFLOPs_per_s_b128",
        "acc_per_100k_params", "train_time_s"]
df[cols].to_csv(RESULTS / "tradeoff_table.csv", index=False)
print(df[cols].T.to_string())

colors = {"MLP": "C0", "CNN": "C1", "Attention": "C2"}
panels = [
    ("params", "Parameters", True),
    ("latency_b1_ms", "Latency, batch 1 (ms, median of 100)", True),
    ("per_sample_b128_ms", "Latency per sample, batch 128 (ms, median of 20)", True),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, (col, label, log) in zip(axes, panels):
    for _, r in df.iterrows():
        ax.scatter(r[col], r["test_acc"], s=120, color=colors[r["model"]], zorder=3)
        ax.annotate(r["model"], (r[col], r["test_acc"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=10)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(label)
    ax.grid(alpha=0.3, which="both")
    ax.margins(x=0.3)
axes[0].set_ylabel("Test accuracy (5 epochs)")
lo, hi = df["test_acc"].min(), df["test_acc"].max()
axes[0].set_ylim(lo - 0.01, hi + 0.01)
fig.suptitle("Accuracy vs cost (up and left is better)")
fig.tight_layout()
fig.savefig(RESULTS / "tradeoffs.png", dpi=150)
print("Saved tradeoffs.png")
