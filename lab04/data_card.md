# Data Card: UCI Adult (Census Income)

## Source
- UCI Machine Learning Repository, "Adult" dataset (Becker and Kohavi, 1996), extracted from the 1994 US Census database.
- Loaded with `fetch_openml(name='adult', version=2, as_frame=True)` (OpenML dataset 1590), which combines the original UCI train and test files.

## License
Creative Commons Attribution 4.0 International (CC BY 4.0), as listed on the UCI repository. Reuse requires attribution.

## Size and structure
- 48,842 rows x 15 columns: 14 features (6 integer, 8 categorical) and a binary label `class` (income >50K or <=50K).
- Positive class (>50K): 23.9% of rows.
- Missing values (NaN, `?` in the original files): workclass 2,799, occupation 2,809, native-country 857.
- 52 exact duplicate rows; 5 groups of identical features with conflicting labels (10 rows); 4 rows where relationship and sex disagree (see `results/label_audit.md`).
- After processing: 34,165 train rows (24 duplicates removed), 14,653 test rows (untouched), 107 features.
- Processed data SHA256: `62edffb015ff7ebfda4c89fb9b8860b65371baf68269b4dc492b9eefc11dbdf5` (details in `results/data_manifest.yaml`).

## Intended use
Teaching and benchmarking tabular classification and data-pipeline practice. Not suitable for real decisions about individuals such as credit, hiring or insurance.

## Known biases
- **Sex imbalance.** The positive rate differs strongly by sex:

| sex | Share of rows | Share earning >50K |
|---|---|---|
| Male | 66.8% | 30.4% |
| Female | 33.2% | 10.9% |

- **Race imbalance.** Most rows are White; some groups are small, so results for them are less reliable:

| race | Share of rows | Share earning >50K |
|---|---|---|
| White | 85.5% | 25.4% |
| Black | 9.6% | 12.1% |
| Asian-Pac-Islander | 3.1% | 26.9% |
| Amer-Indian-Eskimo | 1.0% | 11.7% |
| Other | 0.8% | 12.3% |

- **Geography.** 89.7% of rows are from the United States; 17 of 41 countries have fewer than 50 rows.
- **Time.** Incomes are from 1994 and not adjusted for inflation; the $50K threshold and the labour market are outdated.
- **Sampling.** `fnlwgt` is a census sampling weight, not a personal attribute; the rows are not a simple random sample of the population.
- **Non-random missingness.** 2,799 rows miss both workclass and occupation, and the 10 "Never-worked" rows have no occupation. Mode imputation assigns these people the most common job, which may be wrong.

## Privacy concerns
- No direct identifiers (names, addresses), but several quasi-identifiers: age, sex, race, native-country, education, occupation and marital status. Rare combinations could single out a person if linked with other data.
- Contains sensitive attributes (sex, race, marital status, national origin). Models can learn them or their proxies, so fairness must be checked before any use.
