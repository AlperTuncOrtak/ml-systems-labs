import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Seeds (global rule)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    rows.append({"model": name, "test_accuracy": round(acc, 4)})
    print(f"{name}: {acc:.4f}")

pd.DataFrame(rows).to_csv(RESULTS / "baseline_accuracy.csv", index=False)
print(f"Saved to {RESULTS / 'baseline_accuracy.csv'}")