from common import CONFIG, LAB, RESULTS
import json


def pct(x):
    return f"{100 * x:.1f}%"


def group_table(df, col, pos):
    share = df[col].value_counts(normalize=True)
    rate = pos.groupby(df[col], observed=True).mean()
    rows = [f"| {k} | {pct(share[k])} | {pct(rate[k])} |" for k in share.index]
    return "\n".join([f"| {col} | Share of rows | Share earning >50K |",
                      "|---|---|---|"] + rows)


def write(df, processed_hash):
    t = CONFIG["dataset"]["target"]
    pos = df[t] == CONFIG["dataset"]["positive_label"]
    s = json.loads((RESULTS / "preprocess_summary.json").read_text())
    miss = df.isna().sum()
    both = int((df["workclass"].isna() & df["occupation"].isna()).sum())
    never = df["workclass"] == "Never-worked"
    us = (df["native-country"] == "United-States").mean()
    counts = df["native-country"].value_counts()
    small = int((counts < 50).sum())
    sex_table = group_table(df, "sex", pos)
    race_table = group_table(df, "race", pos)
    text = f"""# Data Card: UCI Adult (Census Income)

## Source
- UCI Machine Learning Repository, "Adult" dataset (Becker and Kohavi, 1996), extracted from the 1994 US Census database.
- Loaded with `fetch_openml(name='adult', version=2, as_frame=True)` (OpenML dataset 1590), which combines the original UCI train and test files.

## License
Creative Commons Attribution 4.0 International (CC BY 4.0), as listed on the UCI repository. Reuse requires attribution.

## Size and structure
- {len(df):,} rows x {df.shape[1]} columns: 14 features (6 integer, 8 categorical) and a binary label `class` (income >50K or <=50K).
- Positive class (>50K): {pct(pos.mean())} of rows.
- Missing values (NaN, `?` in the original files): workclass {miss['workclass']:,}, occupation {miss['occupation']:,}, native-country {miss['native-country']:,}.
- 52 exact duplicate rows; 5 groups of identical features with conflicting labels (10 rows); 4 rows where relationship and sex disagree (see `results/label_audit.md`).
- After processing: {s['train_rows']:,} train rows ({s['duplicates_removed_from_train']} duplicates removed), {s['test_rows']:,} test rows (untouched), {s['processed_features']} features.
- Processed data SHA256: `{processed_hash}` (details in `results/data_manifest.yaml`).

## Intended use
Teaching and benchmarking tabular classification and data-pipeline practice. Not suitable for real decisions about individuals such as credit, hiring or insurance.

## Known biases
- **Sex imbalance.** The positive rate differs strongly by sex:

{sex_table}

- **Race imbalance.** Most rows are White; some groups are small, so results for them are less reliable:

{race_table}

- **Geography.** {pct(us)} of rows are from the United States; {small} of {len(counts)} countries have fewer than 50 rows.
- **Time.** Incomes are from 1994 and not adjusted for inflation; the $50K threshold and the labour market are outdated.
- **Sampling.** `fnlwgt` is a census sampling weight, not a personal attribute; the rows are not a simple random sample of the population.
- **Non-random missingness.** {both:,} rows miss both workclass and occupation, and the {int(never.sum())} "Never-worked" rows have no occupation. Mode imputation assigns these people the most common job, which may be wrong.

## Privacy concerns
- No direct identifiers (names, addresses), but several quasi-identifiers: age, sex, race, native-country, education, occupation and marital status. Rare combinations could single out a person if linked with other data.
- Contains sensitive attributes (sex, race, marital status, national origin). Models can learn them or their proxies, so fairness must be checked before any use.
"""
    (LAB / "data_card.md").write_text(text)
    print("Wrote data_card.md")
