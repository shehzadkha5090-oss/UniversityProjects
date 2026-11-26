# Project 18 — Lab 11 Text Log Feature Engineering

## Dataset
UCI Student Performance dataset (`student-por.csv`, shared copy at
`../shared_dataset/dataset.csv`): 649 rows x 33 columns, semicolon-delimited
(read with `sep=';'`). Only `student-por.csv` was used — not `student-mat.csv`
or any other dataset.

## Target
`pass = (G3 >= 10)`: pass 549 (84.6%), fail 100 (15.4%).

Split: stratified 75/25 train/test, seed 42 (train 486, test 163).

## Caveat
CAVEAT: There are NO raw server logs in this dataset. The text in this project is CONSTRUCTED from structured fields, not real logs, so this demonstrates log-style text feature engineering technique, not real log mining.

## Method
G3 was NEVER used as a feature, and neither G3 nor the word 'pass' (derived
from G3) appears in the constructed text. One short log-style text
description was built per student from that student's actual structured
field values only (no invented facts), for example:

> school GP sex F age 18 address U famsize GT3 Pstatus A Medu 4 Fedu 4 Mjob at_home Fjob teacher reason course guardian mother traveltime 2 studytime 2 failures 0 schoolsup yes famsup no paid no activities no nursery yes higher yes internet no romantic no famrel 4 freetime 3 goout 4 Dalc 1 Walc 1 health 3 absences 4 G1 0 G2 11

Three models were compared on the same split, all Pipelines fit on train only:

- (a) Structured-features LogisticRegression baseline: numeric median-impute +
  StandardScaler, categorical most-frequent-impute + OneHotEncoder.
- (b) TF-IDF (word n-grams 1-2, `min_df=2`) + LogisticRegression on the text only.
- (c) Combined structured + TF-IDF model (ColumnTransformer concatenation,
  then LogisticRegression).

The structured baseline was re-run here for a fair same-split comparison;
no earlier number was copied.

## Results (test set)
- (a) Structured LogisticRegression baseline: ROC-AUC = 0.9420, accuracy = 0.9202
- (b) TF-IDF (word n-grams 1-2, min_df=2) + LogisticRegression (text only): ROC-AUC = 0.8916, accuracy = 0.8712
- (c) Combined structured + TF-IDF LogisticRegression: ROC-AUC = 0.9455, accuracy = 0.9141

## Top TF-IDF terms (text-only model, by LogisticRegression coefficient)
Top 10 POSITIVE coefficients (associated with pass):
- `11`: 1.2477
- `13`: 0.9868
- `12`: 0.9044
- `g2 11`: 0.8566
- `14`: 0.7563
- `school gp`: 0.7126
- `gp`: 0.7126
- `gp sex`: 0.7126
- `failures 0`: 0.6910
- `0 schoolsup`: 0.6910

Top 10 NEGATIVE coefficients (associated with fail):
- `7`: -2.4040
- `8`: -2.1022
- `g2 8`: -1.6398
- `g1 7`: -1.4612
- `7 g2`: -1.4612
- `g2 7`: -1.3750
- `g1 8`: -1.2616
- `8 g2`: -1.2616
- `higher no`: -1.0595
- `no internet`: -1.0595

## Files
- `main.py` — self-contained script (loads the dataset via
  `Path(__file__).resolve().parents[1] / 'shared_dataset' / 'dataset.csv'`)
- `results/results.txt` — full results
- `visualizations/roc_auc_comparison.png` — ROC-AUC comparison bar chart
