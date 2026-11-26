# Project 05 — Lab 5: Custom Cross-Validation

## Dataset
UCI Student Performance (Portuguese course, `student-por.csv`), shared file
`shared_dataset/dataset.csv`, semicolon-delimited, 649 rows × 33 columns
(GP 423, MS 226).

## Target and model
Target: pass = (G3 >= 10). Pass 549 (84.6%), fail 100 (15.4%). G3 is never a
feature; G1/G2 are included by default. Model: LogisticRegression in the
standard pipeline (median/most-frequent imputation, scaling, one-hot).

## Method — manual loops only
`cross_val_score` was NOT used for the main results. The folds were looped
manually, refitting the pipeline inside each fold:
- (a) Shuffled 5-fold KFold (seed 42).
- (b) Stratified 5-fold KFold (seed 42).
- (c) LeaveOneGroupOut grouped by `school` (2 folds: train GP/test MS and
  train MS/test GP).

## Results (from `results/results.txt`, ROC-AUC)
| Scheme | Fold ROC-AUCs | Mean | Std |
|---|---|---|---|
| Shuffled 5-fold KFold | 0.9594, 0.9613, 0.9586, 0.9695, 0.9259 | 0.9549 | 0.0150 |
| Stratified 5-fold KFold | 0.9586, 0.9332, 0.9700, 0.9577, 0.9450 | 0.9529 | 0.0127 |
| LeaveOneGroupOut by school | 0.9374, 0.9215 | 0.9295 | 0.0079 |

## Interpretation
The school-grouped score (0.9295) is lower than the shuffled score (0.9549)
by 0.0254 — a real but modest gap for logistic regression on this data
(the gap is much larger for the RandomForest cross-school runs in Lab 3).
It is lower because each grouped fold trains on one school and tests on a
genuinely different school (GP pass 92.4% vs MS 69.9%), so the distribution
shift hits it in full: the model has never seen the test school's student
mix. The shuffled score is optimistic for deployment to a new school because
shuffling mixes both schools into every train and test fold, so test rows
resemble training rows; it estimates "a new student from a known school",
not "a new school". For deployment to an unseen school, the grouped
LeaveOneGroupOut score is the honest estimate.

## How to run
```
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```
Outputs: `results/results.txt`, `visualizations/cv_comparison.png`.
