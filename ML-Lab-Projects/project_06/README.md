# Lab 6 — Bias & Fairness

**Dataset:** UCI Student Performance, `student-por.csv` (649 rows, 33 columns, semicolon-delimited), shared at `../shared_dataset/dataset.csv`.
**Target:** pass = (G3 >= 10); pass 549 (84.6%), fail 100 (15.4%). G3 is never a feature. Features: all columns except G3.
**Split:** stratified 75/25, seed 42 — train 486, test 163. (Note: 649 × 0.25 rounds up in sklearn, so the test set is 163, not 162.)
**Method:** LogisticRegression baseline vs manual reweighing (sample weights inversely proportional to sex × label frequencies, computed on train only, passed via `sample_weight`). No fairness library was used. Sex was included as a model feature in both models (default feature set).

## Results (test set, copied from `results/results.txt`)

- Real pass rates (full data): F 0.869 (n=383), M 0.812 (n=266).
- Baseline LogisticRegression: ROC-AUC = 0.9420; sex DPD = 0.0134, sex EOD (TPR gap) = 0.0079; school DPD = 0.2664, school EOD = 0.1014.
- Reweighed LogisticRegression: ROC-AUC = 0.9307 (change −0.0113); sex DPD = 0.0098, sex EOD = 0.0013; school DPD = 0.4104, school EOD = 0.3090.

## Discussion

Reweighing by sex slightly narrowed the sex gaps, at a small ROC-AUC cost, but it worsened the school slice (GP vs MS) substantially — so it did not help fairness overall; it only rebalanced the attribute it was weighted on. With only 163 test rows, small gap changes are noisy and are not claimed as proof of fairness.

## Run

```bash
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```

Outputs: `results/results.txt`, `visualizations/fairness_gaps.png`.
