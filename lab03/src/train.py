from common import set_seeds, get_loaders, RESULTS
import time
import pandas as pd
import torch
import torch.nn as nn
from models import get_models

EPOCHS = 5
LR = 1e-3


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += len(y)
    return correct / total


log, final = [], []
for name in ["MLP", "CNN", "Attention"]:
    set_seeds()
    model = get_models()[name]
    train_loader, val_loader, test_loader = get_loaders()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    t_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            n += len(y)
        val_acc = evaluate(model, val_loader)
        log.append({"model": name, "epoch": epoch,
                    "train_loss": round(running / n, 4), "val_acc": round(val_acc, 4)})
        print(f"{name:<9} epoch {epoch} | train loss {running / n:.4f} | val acc {val_acc:.4f}")
    train_time = time.perf_counter() - t_start

    test_acc = evaluate(model, test_loader)
    final.append({"model": name, "test_acc": round(test_acc, 4),
                  "train_time_s": round(train_time, 1)})
    print(f"{name:<9} TEST ACC {test_acc:.4f} | train time {train_time:.1f} s\n")

pd.DataFrame(log).to_csv(RESULTS / "training_log.csv", index=False)
pd.DataFrame(final).to_csv(RESULTS / "test_accuracy.csv", index=False)
print(pd.DataFrame(final).to_string(index=False))
