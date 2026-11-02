# Label-quality audit

## Duplicate-conflict audit (all 48,842 rows)

- Groups of identical feature rows with conflicting income labels: 5
- Rows involved: 10 (listed in `duplicate_conflicts.csv`)

## Data-dictionary checks

| Rule | Violations in 50-row sample | Violations in full data |
|---|---|---|
| age out of range | 0 | 0 |
| hours out of range | 0 | 0 |
| education-num mismatch | 0 | 0 |
| Husband but not Male | 0 | 1 |
| Wife but not Female | 0 | 3 |
| spouse role but not married | 0 | 0 |
| gain and loss both > 0 | 0 | 0 |
| age 17 with graduate degree | 0 | 0 |

Sample rows with no flag: 50/50 (all 50 rows saved in `label_audit_sample50.csv` for manual review).
