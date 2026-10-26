import os
os.environ["OMP_NUM_THREADS"] = "1"  # avoid thread oversubscription (see Lab 1)

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
BATCH = 128


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_datasets():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    full_train = datasets.FashionMNIST(root=DATA, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root=DATA, train=False, download=True, transform=tf)
    train, val = random_split(full_train, [55_000, 5_000],
                              generator=torch.Generator().manual_seed(42))
    return train, val, test


def get_loaders(batch_size=BATCH):
    train, val, test = get_datasets()
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   generator=torch.Generator().manual_seed(42)),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )


if __name__ == "__main__":
    set_seeds()
    train, val, test = get_datasets()
    print(f"train {len(train)} | val {len(val)} | test {len(test)}")
    train_loader, val_loader, test_loader = get_loaders()
    x, y = next(iter(train_loader))
    print("batch x:", tuple(x.shape), x.dtype, "| batch y:", tuple(y.shape))
    print(f"batch mean {x.mean():.3f}, std {x.std():.3f} (expected ~0 and ~1)")
    val_labels = torch.tensor([val[i][1] for i in range(len(val))])
    print("val class counts:", torch.bincount(val_labels).tolist())
    print("batches per epoch:", len(train_loader))
