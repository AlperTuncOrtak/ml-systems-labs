# Lab 1: Environment and First System Measurements

## 1. Goal

Set up a reproducible Python environment with pinned dependencies, train two baseline classifiers on the Breast Cancer Wisconsin dataset, measure their system cost (training time, inference latency, model size, peak memory), and decide which deployment targets (Cloud, Edge, Mobile, TinyML) each model fits.

## 2. Method

- **Environment:** Python 3.11.8 (via pyenv) in an isolated `.venv`, packages pinned in `requirements.txt`. Versions are recorded in `results/versions.txt`.
- **Hardware:** Apple Silicon MacBook Pro (arm64), macOS 26.
- **Data:** `load_breast_cancer` (569 samples, 30 features), 70/30 stratified split with `random_state=42` (398 train / 171 test). Seeds fixed with `random` and `numpy`.
- **Models:** `LogisticRegression(max_iter=1000, random_state=42)` and `RandomForestClassifier(n_estimators=100, random_state=42)`, default settings otherwise, no feature scaling.
- **Training time:** 1 warm-up fit, then median of 5 timed fits (`time.perf_counter`).
- **Inference latency:** single test sample, 1 warm-up, then median of 100 timed predictions.
- **Model size:** fitted model saved with `joblib.dump` (`results/model_logreg.joblib`, `results/model_rf.joblib`).
- **Peak memory:** process RSS sampled with `memory_profiler.memory_usage` during training and during 100 inference calls.

## 3. Results

### Accuracy and system cost

| Metric | LogisticRegression | RandomForest |
|---|---|---|
| Test accuracy | 0.9415 | 0.9357 |
| Training time (median, s) | 0.3514 | 0.0844 |
| Inference latency (median, ms) | 0.0421 | 0.9345 |
| Model size (bytes) | 1,103 | 290,905 |
| Model size (KB) | 1.08 | 284.09 |
| Peak RSS during training (MB) | 142.0 | 85.4 |
| Peak RSS during inference (MB) | 81.2 | 76.0 |
| Training memory delta over baseline (MB) | -0.66 | 0.00 |

Raw data: `results/baseline_accuracy.csv`, `results/system_metrics.csv`.

### Deployment fit

| Target | Budget (memory / latency / size) | LogisticRegression | RandomForest |
|---|---|---|---|
| Cloud | ≥ 1 GB / ≤ 100 ms / ≤ 500 MB | Fits | Fits |
| Edge | 256–1024 MB / ≤ 50 ms / ≤ 50 MB | Fits | Fits |
| Mobile | 64–256 MB / ≤ 20 ms / ≤ 10 MB | Fits | Fits |
| TinyML | ≤ 256 KB / ≤ 10 ms / ≤ 100 KB | Model fits, Python runtime does not | Does not fit (size 284 KB > 100 KB) |

**Justification:** Both models meet the Cloud, Edge and Mobile budgets with a wide margin. The tightest constraint is Mobile memory: the measured process RSS (76–143 MB) stays within the 64–256 MB range, and both latencies are far below 20 ms. For TinyML, the Logistic Regression model is only 31 parameters (30 weights + bias, about 124 bytes as float32), so it would fit on a microcontroller if exported to C, but the Python/scikit-learn process used here needs over 75 MB. The Random Forest model file alone is 284 KB, almost three times the 100 KB limit, so it cannot fit without reducing the number or depth of trees.

**Caveat:** Latencies were measured on a fast laptop CPU. Edge, mobile and especially microcontroller CPUs are slower, so real latencies would be higher. Random Forest's 0.93 ms would still stay under the Mobile limit even with a 10–20x slowdown.

## 4. Conclusions

1. **The two models have almost the same accuracy but very different system costs.** Logistic Regression is about 0.6 percentage points more accurate, about 22x faster at single-sample inference and about 260x smaller on disk. On this dataset the simpler model is the better choice on every axis except training time.
2. **Memory is dominated by the runtime, not the model.** The training memory delta was within measurement noise (≈ 0 MB, slightly negative for one model), while the Python process itself used 76–143 MB. RSS is too coarse to measure models this small. For constrained targets, the runtime matters more than the model itself.
3. **Training time does not predict inference cost.** Logistic Regression trained slower than Random Forest because, without feature scaling, the lbfgs solver hit the 1000-iteration limit (ConvergenceWarning). Inference, however, is a single dot product for LR and 100 tree traversals for RF. Scaling the features would likely make LR training faster too.
