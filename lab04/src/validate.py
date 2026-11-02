from common import LAB, RESULTS
import pandas as pd
import yaml
from pandas.api.types import is_integer_dtype, is_numeric_dtype

SCHEMA = yaml.safe_load((LAB / "schema.yaml").read_text())
MISSING = SCHEMA["missing_token"]


def validate(df):
    """Run schema, range, null, category and duplicate checks.
    Reports violations only: never drops or modifies rows."""
    checks = []

    def add(check, status, detail):
        checks.append({"check": check, "status": status, "detail": detail})

    expected = list(SCHEMA["columns"])
    missing_cols = [c for c in expected if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected]
    add("columns", "PASS" if not missing_cols and not extra_cols else "FAIL",
        f"{df.shape[1]} columns; missing={missing_cols}; extra={extra_cols}")

    for col, spec in SCHEMA["columns"].items():
        if col not in df.columns:
            continue
        s = df[col]
        if spec["type"] == "int":
            add(f"type: {col}", "PASS" if is_integer_dtype(s) else "FAIL",
                f"expected int, got {s.dtype}")
            if "range" in spec and is_numeric_dtype(s):
                lo, hi = spec["range"]
                bad = int(((s < lo) | (s > hi)).sum())
                add(f"range: {col}", "PASS" if bad == 0 else "FAIL",
                    f"allowed [{lo}, {hi}], observed [{s.min()}, {s.max()}], "
                    f"{bad} violations")
        else:
            is_cat = isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object
            add(f"type: {col}", "PASS" if is_cat else "FAIL",
                f"expected category, got {s.dtype}")
            values = s.astype(object)
            unknown = sorted(set(values.dropna().unique())
                             - set(spec["allowed"]) - {MISSING})
            add(f"categories: {col}", "PASS" if not unknown else "FAIL",
                f"{len(unknown)} values outside the data dictionary"
                + (f": {unknown[:5]}" if unknown else ""))
            n_missing = int(values.isna().sum() + (values == MISSING).sum())
            if n_missing:
                add(f"missing: {col}", "WARN",
                    f"{n_missing} missing ({n_missing / len(df):.2%}), "
                    f"NaN or '{MISSING}'; kept, imputed later from train")

    dup = int(df.duplicated().sum())
    add("duplicates", "WARN" if dup else "PASS",
        f"{dup} exact duplicate rows; kept here, removed from train only")

    edu = df["education"].astype(object).map(SCHEMA["education_num"])
    mismatch = int((edu.notna() & (edu != df["education-num"])).sum())
    add("consistency: education vs education-num",
        "PASS" if mismatch == 0 else "FAIL", f"{mismatch} rows disagree")

    return pd.DataFrame(checks)


def write_report(report, n_rows):
    counts = report["status"].value_counts().to_dict()
    lines = ["# Validation report", "",
             f"Rows checked: {n_rows}. No rows were dropped or modified.", "",
             f"Summary: {counts.get('PASS', 0)} PASS, {counts.get('WARN', 0)} WARN, "
             f"{counts.get('FAIL', 0)} FAIL", "",
             "| Check | Status | Detail |", "|---|---|---|"]
    lines += [f"| {r.check} | {r.status} | {r.detail} |" for r in report.itertuples()]
    (RESULTS / "validation_report.md").write_text("\n".join(lines) + "\n")
    report.to_csv(RESULTS / "validation_report.csv", index=False)


if __name__ == "__main__":
    from ingest import ingest
    df = ingest()
    report = validate(df)
    write_report(report, len(df))
    pd.set_option("display.max_colwidth", 90)
    print(report.to_string(index=False))
    print("\nSummary:", report["status"].value_counts().to_dict())
