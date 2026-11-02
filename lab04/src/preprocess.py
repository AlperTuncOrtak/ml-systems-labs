from common import CONFIG, DATA, RESULTS, set_seeds
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from ingest import ingest
from validate import MISSING, validate, write_report
from transformers import PercentileClipper, add_features

TARGET = CONFIG["dataset"]["target"]
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
DERIVED = ["capital_net", "hours_per_age"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "sex", "native-country"]


def load_validated():
    df = ingest()
    report = validate(df)
    write_report(report, len(df))
    if (report["status"] == "FAIL").any():
        raise ValueError("Validation failed, see results/validation_report.md")
    X = df.drop(columns=TARGET)
    X[CATEGORICAL] = X[CATEGORICAL].astype(object).replace(MISSING, np.nan)
    y = (df[TARGET] == CONFIG["dataset"]["positive_label"]).astype(int)
    return X, y


def split_and_dedupe(X, y):
    sp = CONFIG["split"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=sp["test_size"], stratify=y, random_state=sp["random_state"])
    dup = X_tr.assign(_y=y_tr).duplicated().values  # exact duplicates, train only
    return X_tr[~dup], X_te, y_tr[~dup], y_te, int(dup.sum()), len(X_tr)


def build_preprocessor(with_features):
    c = CONFIG["cleaning"]
    numeric = NUMERIC + (DERIVED if with_features else [])
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy=c["numeric_impute"])),
        ("clip", PercentileClipper(c["clip_lower_pct"], c["clip_upper_pct"])),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy=c["categorical_impute"])),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([("num", num_pipe, numeric),
                                 ("cat", cat_pipe, CATEGORICAL)])
    steps = [("features", FunctionTransformer(add_features))] if with_features else []
    return Pipeline(steps + [("columns", columns)])


def run():
    set_seeds()
    X, y = load_validated()
    X_tr, X_te, y_tr, y_te, n_dup, n_tr_before = split_and_dedupe(X, y)

    rows, fitted = [], {}
    for name, feats in [("without_derived", False), ("with_derived", True)]:
        pre = build_preprocessor(feats).fit(X_tr)          # fit on train only
        Xtr_p, Xte_p = pre.transform(X_tr), pre.transform(X_te)
        clf = LogisticRegression(max_iter=1000, random_state=CONFIG["seed"])
        pred = clf.fit(Xtr_p, y_tr).predict(Xte_p)
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        rows.append({"variant": name, "n_features": Xtr_p.shape[1],
                     "accuracy": round(accuracy_score(y_te, pred), 4),
                     "f1": round(f1_score(y_te, pred), 4),
                     "TP": tp, "FP": fp, "FN": fn, "TN": tn})
        fitted[name] = (pre, Xtr_p, Xte_p)

    pre, Xtr_p, Xte_p = fitted["with_derived"]
    joblib.dump(pre, RESULTS / "preprocessor.joblib")
    clip = pre.named_steps["columns"].named_transformers_["num"].named_steps["clip"]
    pd.DataFrame({"feature": NUMERIC + DERIVED, "p1_train": clip.low_,
                  "p99_train": clip.high_}).to_csv(RESULTS / "clip_limits.csv", index=False)

    out = DATA / "processed"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_train.npy", Xtr_p)
    np.save(out / "X_test.npy", Xte_p)
    np.save(out / "y_train.npy", y_tr.to_numpy())
    np.save(out / "y_test.npy", y_te.to_numpy())

    summary = {"raw_rows": len(X), "train_rows_before_dedupe": n_tr_before,
               "duplicates_removed_from_train": n_dup, "train_rows": len(X_tr),
               "test_rows": len(X_te), "processed_features": int(Xtr_p.shape[1]),
               "positive_rate_train": round(float(y_tr.mean()), 4),
               "positive_rate_test": round(float(y_te.mean()), 4)}
    (RESULTS / "preprocess_summary.json").write_text(json.dumps(summary, indent=2))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "baseline_metrics.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))
    print(pd.read_csv(RESULTS / "clip_limits.csv").to_string(index=False))


if __name__ == "__main__":
    run()
