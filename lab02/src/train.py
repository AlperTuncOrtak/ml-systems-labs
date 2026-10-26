from common import set_seeds, load_data
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nn import init_params, forward, cross_entropy, backward

EPOCHS = 50
BATCH = 32
LRS = [0.001, 0.01, 0.1]
WIDTHS = [16, 64]
RESULTS = Path(__file__).resolve().parents[1] / "results"

X_train, X_test, Y_train, Y_test, y_train, y_test = load_data()


def accuracy(params, X, y):
    return float((forward(params, X)[0].argmax(axis=1) == y).mean())


def train(width, lr):
    set_seeds()  # same init and shuffling order for every run
    params = init_params(width)
    log = []
    for epoch in range(1, EPOCHS + 1):
        perm = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), BATCH):
            b = perm[i:i + BATCH]
            _, cache = forward(params, X_train[b])
            grads = backward(params, cache, Y_train[b])
            for k in params:
                params[k] -= lr * grads[k]
        log.append({
            "width": width, "lr": lr, "epoch": epoch,
            "train_loss": cross_entropy(forward(params, X_train)[0], Y_train),
            "test_loss": cross_entropy(forward(params, X_test)[0], Y_test),
            "test_acc": accuracy(params, X_test, y_test),
        })
        if epoch == 1 or epoch % 10 == 0:
            r = log[-1]
            print(f"W={width:>2} lr={lr:<5} epoch {epoch:>2} | train loss "
                  f"{r['train_loss']:.4f} | test loss {r['test_loss']:.4f} | "
                  f"test acc {r['test_acc']:.4f}")
    return log


logs = []
for width in WIDTHS:
    for lr in LRS:
        logs += train(width, lr)

df = pd.DataFrame(logs)
df.to_csv(RESULTS / "training_log.csv", index=False)

final = df[df["epoch"] == EPOCHS][["width", "lr", "train_loss", "test_loss", "test_acc"]]
final = final.round(4)
final.to_csv(RESULTS / "final_accuracy.csv", index=False)
print("\nFinal results after", EPOCHS, "epochs:")
print(final.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, width in zip(axes, WIDTHS):
    for lr in LRS:
        run = df[(df["width"] == width) & (df["lr"] == lr)]
        line, = ax.plot(run["epoch"], run["train_loss"], label=f"lr={lr} train")
        ax.plot(run["epoch"], run["test_loss"], "--", color=line.get_color(),
                label=f"lr={lr} test")
    ax.axhline(np.log(10), color="gray", lw=0.8, ls=":", label="ln(10)")
    ax.set_title(f"Hidden width W = {width}")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Cross-entropy loss")
axes[1].legend(fontsize=8)
fig.suptitle("Loss curves (solid = train, dashed = test)")
fig.tight_layout()
fig.savefig(RESULTS / "loss_curves.png", dpi=150)
print("Saved loss_curves.png")
