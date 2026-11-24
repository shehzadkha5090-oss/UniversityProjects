# Project 01 — Lab 1: Data Leakage

## Dataset
UCI Student Performance (Portuguese course, `student-por.csv`), shared file
`shared_dataset/dataset.csv`, semicolon-delimited, 649 rows × 33 columns.

## Target and split
Target: pass = (G3 >= 10). Pass 549 (84.6%), fail 100 (15.4%).
Split: stratified 75/25, seed 42 — train 486, test 163.

## Method
Three RandomForest (200 trees) models on the same split:
- (a) TRULY LEAKY: includes G3 itself as a feature (the mistake, shown only to demonstrate leakage).
- (b) End-of-term: G1 + G2, no G3.
- (c) Early-warning: no G1, G2, G3 — only features known at the start of the year.

## Results (from `results/results.txt`, test set of 163)
| Model | ROC-AUC | Accuracy |
|---|---|---|
| (a) Leaky (includes G3) | 1.0000 | 1.0000 |
| (b) End-of-term (G1+G2) | 0.9432 | 0.9141 |
| (c) Early-warning (no G1/G2/G3) | 0.7348 | 0.8282 |

Leakage inflation (a minus c): 0.2652 ROC-AUC.

## Interpretation
G3 is the final grade itself, so model (a) is not predicting at all — it reads
the answer, which is why it scores a perfect 1.0000. G1/G2 are earlier-period
grades, so model (b) is legitimate only at the end of term, after those grades
exist. For an early-warning model that must flag at-risk students at the start
of the year, G1/G2 do not exist yet, so only model (c) is usable. The
corrected, recommended pipeline for early warning is (c), even though its
ROC-AUC (0.7348) is lower — the higher scores of (a) and, for early warning,
(b) are not honestly available at prediction time.

## How to run
```
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```
Outputs: `results/results.txt`, `visualizations/roc_auc_comparison.png`.
