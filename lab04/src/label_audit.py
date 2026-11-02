from common import CONFIG, LAB, RESULTS
import pandas as pd
import yaml
from ingest import ingest

TARGET = CONFIG["dataset"]["target"]
SCHEMA = yaml.safe_load((LAB / "schema.yaml").read_text())
MARRIED = {"Married-civ-spouse", "Married-AF-spouse"}


def dictionary_flags(df):
    """Rule checks against the UCI data dictionary. Returns one text per row."""
    s = df.astype({c: object for c in df.select_dtypes("category").columns})
    rules = {
        "age out of range": ~s["age"].between(17, 90),
        "hours out of range": ~s["hours-per-week"].between(1, 99),
        "education-num mismatch":
            s["education"].map(SCHEMA["education_num"]) != s["education-num"],
        "Husband but not Male": (s["relationship"] == "Husband") & (s["sex"] != "Male"),
        "Wife but not Female": (s["relationship"] == "Wife") & (s["sex"] != "Female"),
        "spouse role but not married":
            s["relationship"].isin(["Husband", "Wife"]) & ~s["marital-status"].isin(MARRIED),
        "gain and loss both > 0": (s["capital-gain"] > 0) & (s["capital-loss"] > 0),
        "age 17 with graduate degree":
            (s["age"] <= 17) & s["education"].isin(["Masters", "Doctorate", "Prof-school"]),
    }
    flags = pd.DataFrame(rules)
    return flags, flags.apply(lambda r: "; ".join(r.index[r]), axis=1)


def duplicate_conflicts(df):
    features = [c for c in df.columns if c != TARGET]
    labels = df.groupby(features, dropna=False, observed=True)[TARGET].nunique()
    conflict_groups = labels[labels > 1]
    key = df[features].astype(str).agg("|".join, axis=1)
    conflict_keys = set(df.loc[conflict_groups.index.size and
                               df.set_index(features).index.isin(conflict_groups.index),
                               features].astype(str).agg("|".join, axis=1))
    rows = df[key.isin(conflict_keys)]
    return len(conflict_groups), rows


def run():
    df = ingest()
    n_groups, conflict_rows = duplicate_conflicts(df)
    conflict_rows.to_csv(RESULTS / "duplicate_conflicts.csv", index=False)

    flags_all, _ = dictionary_flags(df)
    full_counts = flags_all.sum().astype(int)

    sample = df.sample(50, random_state=CONFIG["seed"])
    flags_s, text_s = dictionary_flags(sample)
    sample = sample.assign(flags=text_s.replace("", "OK"))
    sample.to_csv(RESULTS / "label_audit_sample50.csv")

    lines = ["# Label-quality audit", "",
             "## Duplicate-conflict audit (all 48,842 rows)", "",
             f"- Groups of identical feature rows with conflicting income labels: {n_groups}",
             f"- Rows involved: {len(conflict_rows)} (listed in `duplicate_conflicts.csv`)", "",
             "## Data-dictionary checks", "",
             "| Rule | Violations in 50-row sample | Violations in full data |",
             "|---|---|---|"]
    lines += [f"| {r} | {int(flags_s[r].sum())} | {int(full_counts[r])} |"
              for r in flags_s.columns]
    lines += ["", f"Sample rows with no flag: {(sample['flags'] == 'OK').sum()}/50 "
                  "(all 50 rows saved in `label_audit_sample50.csv` for manual review)."]
    (RESULTS / "label_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    flagged = sample[sample["flags"] != "OK"]
    if len(flagged):
        print("\nFlagged sample rows:")
        print(flagged[["age", "education", "education-num", "marital-status",
                       "relationship", "sex", "capital-gain", "capital-loss",
                       "flags"]].to_string())


if __name__ == "__main__":
    run()
