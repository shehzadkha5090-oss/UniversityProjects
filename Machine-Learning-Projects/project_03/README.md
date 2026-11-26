# Project 03 — Lab 3: Distribution Shift

## Dataset
UCI Student Performance (Portuguese course, `student-por.csv`), shared file
`shared_dataset/dataset.csv`, semicolon-delimited, 649 rows × 33 columns.

## Target
Target: pass = (G3 >= 10). G3 is never a feature; G1/G2 are included by default.

## The shift
- School GP: 423 students, pass rate 92.4% (391/423).
- School MS: 226 students, pass rate 69.9% (158/226).

Models are trained on one school and tested on the other, and vice versa
(RandomForest, 200 trees).

## How shift and stability were measured
For each feature, GP vs MS was compared with the KS statistic (numeric
features, plus PSI on 10 quantile bins from the training school) or the
maximum category-proportion difference (categorical features). Stability was
judged in advance as KS < 0.20 (numeric) or proportion difference < 0.20
(categorical). This gave 24 stable features out of 31 (excluding `school`
and `G3`): age, studytime, failures, famrel, freetime, goout, Dalc, Walc,
health, absences, sex, famsize, Pstatus, Mjob, Fjob, reason, guardian,
schoolsup, famsup, paid, activities, nursery, higher, romantic. Notably,
G1 (KS = 0.321), G2 (KS = 0.277), Medu, Fedu, traveltime, address and
internet were judged unstable and dropped from the stable-only model.
Full per-feature KS/PSI values are in `results/results.txt`.

## Results (from `results/results.txt`, held-out-school ROC-AUC)
| Direction | All features | Stable only | Change |
|---|---|---|---|
| Train GP → test MS (423 → 226) | 0.9453 | 0.6851 | −0.2602 |
| Train MS → test GP (226 → 423) | 0.9734 | 0.7712 | −0.2022 |

## Honest verdict
**Stable-only retraining did NOT help in either direction.** Dropping the
shifted features removed the strongest signals (G1/G2) and performance fell
sharply. Stability filtering is not a free fix for distribution shift: when
the shifted features are also the most predictive, removing them costs more
than the shift itself. For deployment to a new school, the cross-school
scores — not a within-school split — are the honest estimate.

## How to run
```
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```
Outputs: `results/results.txt`, `visualizations/cross_school_roc_auc.png`.
