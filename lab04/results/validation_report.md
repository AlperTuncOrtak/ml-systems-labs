# Validation report

Rows checked: 48842. No rows were dropped or modified.

Summary: 30 PASS, 4 WARN, 0 FAIL

| Check | Status | Detail |
|---|---|---|
| columns | PASS | 15 columns; missing=[]; extra=[] |
| type: age | PASS | expected int, got int64 |
| range: age | PASS | allowed [17, 90], observed [17, 90], 0 violations |
| type: workclass | PASS | expected category, got category |
| categories: workclass | PASS | 0 values outside the data dictionary |
| missing: workclass | WARN | 2799 missing (5.73%), NaN or '?'; kept, imputed later from train |
| type: fnlwgt | PASS | expected int, got int64 |
| type: education | PASS | expected category, got category |
| categories: education | PASS | 0 values outside the data dictionary |
| type: education-num | PASS | expected int, got int64 |
| type: marital-status | PASS | expected category, got category |
| categories: marital-status | PASS | 0 values outside the data dictionary |
| type: occupation | PASS | expected category, got category |
| categories: occupation | PASS | 0 values outside the data dictionary |
| missing: occupation | WARN | 2809 missing (5.75%), NaN or '?'; kept, imputed later from train |
| type: relationship | PASS | expected category, got category |
| categories: relationship | PASS | 0 values outside the data dictionary |
| type: race | PASS | expected category, got category |
| categories: race | PASS | 0 values outside the data dictionary |
| type: sex | PASS | expected category, got category |
| categories: sex | PASS | 0 values outside the data dictionary |
| type: capital-gain | PASS | expected int, got int64 |
| range: capital-gain | PASS | allowed [0, 99999], observed [0, 99999], 0 violations |
| type: capital-loss | PASS | expected int, got int64 |
| range: capital-loss | PASS | allowed [0, 4356], observed [0, 4356], 0 violations |
| type: hours-per-week | PASS | expected int, got int64 |
| range: hours-per-week | PASS | allowed [1, 99], observed [1, 99], 0 violations |
| type: native-country | PASS | expected category, got category |
| categories: native-country | PASS | 0 values outside the data dictionary |
| missing: native-country | WARN | 857 missing (1.75%), NaN or '?'; kept, imputed later from train |
| type: class | PASS | expected category, got category |
| categories: class | PASS | 0 values outside the data dictionary |
| duplicates | WARN | 52 exact duplicate rows; kept here, removed from train only |
| consistency: education vs education-num | PASS | 0 rows disagree |
