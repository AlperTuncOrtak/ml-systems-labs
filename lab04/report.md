# Lab 4: Data Pipeline and Quality

Run everything with one command: `python3 lab04/src/run_pipeline.py`

## 1. Goal and problem definition

**Problem.** Predict whether a person's yearly income is above $50K from 14 census attributes (age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country), using the UCI Adult dataset (48,842 rows).

**Output.** A binary label: income >50K (positive) or <=50K.

**Success metrics.** F1 score on the positive class and accuracy, both on a held-out test set. Accuracy alone is misleading here: only 23.9% of people earn >50K, so a model that always predicts <=50K would reach 76.1% accuracy with an F1 of 0. F1 shows how well the minority class is found.

**Constraints.**
- No leakage: the test set must not influence any cleaning or feature decision. The data is split before cleaning, and every statistic (imputation values, clipping limits, scaling) is learned from the training set only.
- Reproducibility: the full pipeline runs with one command, uses fixed seeds and a versioned config, and produces the same processed data every time, verified with a SHA256 hash.

The goal of the lab is not the best model but a trustworthy pipeline: validated input, leakage-free processing, documented data quality and a documented dataset.

## 2. Method

The pipeline is split into steps, each in its own script in `src/`, with all parameters in `config.yaml` and the schema in `schema.yaml`.

1. **Ingestion** (`ingest.py`): loads the data with `fetch_openml(name='adult', version=2, as_frame=True)` without changing it and logs its shape (`results/ingestion_log.txt`).
2. **Validation** (`validate.py`): checks columns, types, ranges (age 17–90, hours 1–99, capital-gain 0–99999, capital-loss 0–4356), allowed categories from the UCI data dictionary, missing values, duplicates, and education vs education-num consistency. In the OpenML version, missing values are already stored as NaN instead of `?`; the null policy treats both as missing. Violations are reported, never silently dropped. The pipeline stops on any FAIL.
3. **Cleaning and transformation** (`preprocess.py`): 70/30 stratified split (`random_state=42`) before any cleaning, then exact duplicates are removed from train only. A scikit-learn pipeline fitted on train only applies median imputation, clipping to the train 1st/99th percentiles and standard scaling to numeric features, and mode imputation plus one-hot encoding to categorical features. Scaling is not in the lab spec but is needed for Logistic Regression to converge on unscaled features like `fnlwgt`; it is also learned from train only. The fitted pipeline is saved to `results/preprocessor.joblib`.
4. **Derived features**: `capital_net = capital-gain − capital-loss` and `hours_per_age = hours-per-week / (age − 16)`. Both are computed per row from that row's own values, so they cannot leak information between rows. Evaluated with `LogisticRegression(max_iter=1000, random_state=42)` with and without them.
5. **Label-quality audit** (`label_audit.py`): duplicate-conflict audit on all rows, and data-dictionary rules checked on 50 random rows (`random_state=42`) and on the full data.
6. **Hash, manifest and data card** (`finalize.py`, `data_card.py`): SHA256 of the processed train/test arrays, stored in `results/data_manifest.yaml` together with the config, schema hash, raw-data hash and package versions. `data_card.md` is generated from the data itself.

## 3. Results

### Validation (`results/validation_report.md`)

30 PASS, 4 WARN, 0 FAIL. All types, ranges and categories match the schema, and education agrees with education-num in every row. Warnings (reported, rows kept):

| Check | Finding |
|---|---|
| Missing workclass | 2,799 rows (5.73%) |
| Missing occupation | 2,809 rows (5.75%) |
| Missing native-country | 857 rows (1.75%) |
| Exact duplicates | 52 rows |

### Cleaning (`results/preprocess_summary.json`, `results/clip_limits.csv`)

| Item | Value |
|---|---|
| Train rows before / after duplicate removal | 34,189 / 34,165 (24 removed) |
| Test rows (untouched) | 14,653 |
| Positive rate train / test | 23.94% / 23.93% |
| Features after one-hot encoding | 107 (105 without derived features) |

Clipping limits learned on train (examples): age [17, 74], hours-per-week [8, 80], capital-gain [0, 15,024], capital-loss [0, 1,980].

### Derived features (`results/baseline_metrics.csv`)

| Variant | Accuracy | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|
| Without derived features | 0.8540 | 0.6601 | 2,078 | 712 | 1,428 | 10,435 |
| With derived features | **0.8578** | **0.6696** | 2,112 | 690 | 1,394 | 10,457 |

Check of the "with" row from its confusion matrix: precision = 2112 / (2112 + 690) = 0.754, recall = 2112 / (2112 + 1394) = 0.602, F1 = 2 × 0.754 × 0.602 / (0.754 + 0.602) = 0.670, accuracy = (2112 + 10457) / 14653 = 0.858.

### Label quality (`results/label_audit.md`)

| Check | Result |
|---|---|
| Duplicate-conflict groups (same features, different label) | 5 groups, 10 rows |
| 50 random rows vs data dictionary | 50/50 with no violation |
| Full data: relationship "Husband" but sex not Male | 1 row |
| Full data: relationship "Wife" but sex not Female | 3 rows |
| Full data: other rules (ranges, education, marital status, gain and loss together) | 0 rows |

### Reproducibility (`results/reproducibility_log.txt`, `results/data_manifest.yaml`)

The pipeline was run twice with one command. Both runs produced the same processed-data hash, `62edffb015ff7ebfda4c89fb9b8860b65371baf68269b4dc492b9eefc11dbdf5`, and the second run logged `MATCH`.

### Data card

See `data_card.md`: source, CC BY 4.0 license, size, known biases (for example, 30.4% of men vs 10.9% of women earn >50K) and privacy concerns.

## 4. Conclusions

1. **The order of the pipeline is what prevents leakage.** Splitting before cleaning and fitting every statistic on train only means the test set is treated like truly unseen data. If the clipping limits or imputation values had been computed on all 48,842 rows, the test set would have influenced the training process and the reported scores would be slightly optimistic without any visible error.
2. **Validation should report, not silently fix.** All checks passed, but the warnings carry real information: missing values are concentrated in workclass and occupation together, and 52 duplicates exist. A full-data rule check also found 4 rows where relationship and sex contradict each other, which the 50-row sample missed. Small samples are useful for manual review but cannot replace automated checks on the full data.
3. **A derived feature helps only if it adds information the model cannot build itself.** Adding the two features improved F1 from 0.660 to 0.670 and accuracy from 0.854 to 0.858. `hours_per_age` is a ratio, which a linear model cannot form from its inputs, while `capital_net` is a linear combination of features the model already has. The gain is small and measured on a single split, so it should be confirmed with cross-validation before relying on it.
