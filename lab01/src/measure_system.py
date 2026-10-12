import random
import statistics
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from memory_profiler import memory_usage
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)
# torch is not imported here: it is unused in this lab and would add
# ~100+ MB to process RSS, distorting the memory measurements.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def timed_fit(model, X, y):
    m = clone(model)
    t0 = time.perf_counter()
    m.fit(X, y)
    return time.perf_counter() - t0


def predict_loop(model, sample, n=100):
    for _ in range(n):
        model.predict(sample)


def measure(name, model, X_train, y_train, X_test, y_test):
    timed_fit(model, X_train, y_train)
    train_times = [timed_fit(model, X_train, y_train) for _ in range(5)]

    base_mem = memory_usage(-1, interval=0.01, timeout=0.2, max_usage=True)
    fitted = clone(model)
    peak_train = memory_usage((fitted.fit, (X_train, y_train)),
                              interval=0.001, max_usage=True)

    sample = X_test[:1]
    fitted.predict(sample)
    lat_ms = []
    for _ in range(100):
        t0 = time.perf_counter()
        fitted.predict(sample)
        lat_ms.append((time.perf_counter() - t0) * 1000)

    peak_infer = memory_usage((predict_loop, (fitted, sample)),
                              interval=0.001, max_usage=True)

    path = RESULTS / f"model_{name}.joblib"
    joblib.dump(fitted, path)
    size_b = path.stat().st_size

    return {
        "model": name,
        "test_accuracy": round(accuracy_score(y_test, fitted.predict(X_test)), 4),
        "train_time_median_s": round(statistics.median(train_times), 4),
        "inference_median_ms": round(statistics.median(lat_ms), 4),
        "model_size_bytes": size_b,
        "model_size_kb": round(size_b / 1024, 2),
        "baseline_rss_mb": round(base_mem, 1),
        "peak_rss_train_mb": round(peak_train, 1),
        "peak_rss_infer_mb": round(peak_infer, 1),
        "train_mem_delta_mb": round(peak_train - base_mem, 2),
    }


if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    models = {
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    rows = [measure(n, m, X_train, y_train, X_test, y_test)
            for n, m in models.items()]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "system_metrics.csv", index=False)
    print(df.T.to_string())
