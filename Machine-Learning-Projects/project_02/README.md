# Project 02 — Lab 2: Missing Data

## Dataset
UCI Student Performance (Portuguese course, `student-por.csv`), shared file
`shared_dataset/dataset.csv`, semicolon-delimited, 649 rows × 33 columns.

## Target and split
Target: pass = (G3 >= 10). Pass 549 (84.6%), fail 100 (15.4%).
Split: stratified 75/25, seed 42 — train 486, test 163.

## Real missingness (verified from the file)
**0 missing values in the whole file.** Every one of the 33 columns is
649/649 complete (0.0% missing). There is therefore nothing real to impute.

## Labelled simulations (introduced by `main.py`, NOT original data)
- **MCAR:** 20% of feature values masked at random — 4,165 missing cells (20.1% of feature cells).
- **MAR:** missingness in `absences` and `studytime` depends on observed `school` only (40% if MS, 5% otherwise) — 216 missing cells (1.0%).
- **MNAR:** missingness in `absences` depends on its own unobserved value (50% if absences > 10, else 5%) and in `studytime` (40% if studytime == 1, else 5%) — 148 missing cells (0.7%).

## Results (from `results/results.txt`, RandomForest, test ROC-AUC)
| Simulation | Median/mode imputation | KNN imputation (k=5) |
|---|---|---|
| MCAR | 0.9319 | 0.9280 |
| MAR | 0.9451 | 0.9446 |
| MNAR | 0.9442 | 0.9423 |

KNN vs median/mode differences are small here because G1/G2 carry most
of the signal and are never masked in the MAR/MNAR simulations. MNAR remains
the hardest case in principle, since the missingness depends on values that
are themselves hidden and no imputer can fully recover them.

## How to run
```
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```
Outputs: `results/results.txt`, `visualizations/imputation_comparison.png`.
