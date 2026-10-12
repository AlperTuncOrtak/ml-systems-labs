# Lab 1: Environment and First System Measurements

## 1. Goal

Set up a reproducible Python environment with pinned dependencies, train two baseline classifiers on the Breast Cancer Wisconsin dataset, measure their system cost (training time, inference latency, model size, peak memory), and decide which deployment targets (Cloud, Edge, Mobile, TinyML) each model fits.

## 2. Method

- **Environment:** Python 3.11.8 (via pyenv) in an isolated `.venv`, packages pinned in `requirements.txt`. Versions are recorded in `results/versions.txt`.
- **Hardware:** Apple Silicon MacBook Pro (arm64), macOS 26.
- **Data:** `load_breast_cancer` (569 samples, 30 features), 70/30 stratified split with `random_state=42` (398 train / 171 test).
- **Seeds:** `random`, `numpy`, `torch` and `torch.cuda` all seeded with 42.
- **Threads:** `OMP_NUM_THREADS=1` is set before any import. Without it, importing torch caused thread oversubscription and Logistic Regression training slowed from 0.03 s to 4.53 s (see Conclusion 3).
- **Models:** `LogisticRegression(max_iter=1000, random_state=42)` and `RandomForestClassifier(n_estimators=100, random_state=42)`, default settings otherwise, no feature scaling.
- **Training time:** 1 warm-up fit, then median of 5 timed fits (`time.perf_counter`).
- **Inference latency:** single test sample, 1 warm-up, then median of 100 timed predictions.
- **Model size:** each fitted model saved with `joblib.dump` to `results/model.joblib` and its size recorded. The final `results/model.joblib` contains both fitted models.
- **Peak memory:** process RSS sampled with `memory_profiler.memory_usage` during training and during 100 inference calls.
- **Repeatability:** the full measurement script was run 3 times. Single values below are from the last run (saved in `results/system_metrics.csv`), ranges cover all 3 runs.

## 3. Results

### Accuracy and system cost

| Metric | LogisticRegression | RandomForest |
|---|---|---|
| Test accuracy | 0.9415 | 0.9357 |
| Training time, median (s) | 0.0326 (range 0.0292–0.0326) | 0.0601 (range 0.0589–0.0637) |
| Inference latency, median (ms) | 0.0191 (range 0.0171–0.0191) | 0.9161 (range 0.8494–0.9828) |
| Model size (bytes) | 1,103 | 290,905 |
| Model size (KB) | 1.08 | 284.09 |
| Baseline process RSS (MB) | 195.5 (range 195.5–290.5) | 111.8 (range 100.0–130.3) |
| Peak RSS during training (MB) | 191.7 (range 191.7–290.5) | 111.8 (range 101.2–130.3) |
| Peak RSS during inference (MB) | 188.0 (range 126.4–209.3) | 210.8 (range 209.4–213.5) |

Raw data: `results/baseline_accuracy.csv`, `results/system_metrics.csv`.

### Deployment fit

| Target | Budget (memory / latency / size) | LogisticRegression | RandomForest |
|---|---|---|---|
| Cloud | ≥ 1 GB / ≤ 100 ms / ≤ 500 MB | Fits | Fits |
| Edge | 256–1024 MB / ≤ 50 ms / ≤ 50 MB | Fits | Fits |
| Mobile | 64–256 MB / ≤ 20 ms / ≤ 10 MB | Borderline: process RSS exceeded 256 MB in 2 of 3 runs | Fits (RSS ≤ 214 MB) |
| TinyML | ≤ 256 KB / ≤ 10 ms / ≤ 100 KB | Model fits, Python runtime does not | Does not fit (size 284 KB > 100 KB) |

**Justification:** Both models meet the Cloud and Edge budgets on every axis: the highest measured RSS (290.5 MB) is well below 1024 MB, latencies are under 1 ms, and both files are far below 50 MB. For Mobile, latency (≤ 1 ms vs 20 ms) and size (≤ 0.3 MB vs 10 MB) fit easily, but process memory is borderline. The measured RSS (100–291 MB) is almost entirely the Python runtime and the libraries loaded into it, including torch, which is imported only for seeding and would not be shipped in a real deployment. The models themselves add no measurable memory. For TinyML, Logistic Regression is only 31 parameters (30 weights + bias, about 124 bytes as float32), so it would fit on a microcontroller if exported to C, but not as a Python/scikit-learn process. The Random Forest file alone is 284 KB, almost three times the 100 KB limit, so it cannot fit without fewer or shallower trees.

**Caveat:** Latencies were measured on a fast laptop CPU. Edge, mobile and especially microcontroller CPUs are slower, so real latencies would be higher. Random Forest's ~0.9 ms would still stay under the Mobile limit even with a 10–20x slowdown.

## 4. Conclusions

1. **The simpler model wins on every system metric at almost the same accuracy.** Logistic Regression is 0.58 percentage points more accurate, about 2x faster to train, about 50x faster at single-sample inference and about 264x smaller on disk than Random Forest.
2. **Process RSS cannot measure models this small.** The training memory delta was within noise, and the same script reported baseline RSS anywhere between 100 and 291 MB across runs. The runtime, not the model, decides the memory budget, and a single RSS reading is not enough to make a deployment decision near a budget boundary.
3. **The environment can matter more than the model.** Importing torch without limiting threads made Logistic Regression training about 150x slower (4.53 s vs 0.03 s) with identical code and data. Setting `OMP_NUM_THREADS=1` fixed it and made training faster than the original torch-free run (0.35 s). System measurements are only meaningful if the thread configuration is controlled and reported.
