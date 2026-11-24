# Lab 12 — Calibration

**Dataset:** UCI Student Performance, `student-por.csv` (649 rows), shared at `../shared_dataset/dataset.csv`.
**Target:** pass = (G3 >= 10). G3 is never a feature.
**Split:** stratified, seed 42 — train 388, calibration 98, test 163 (nominal 60/15/25, rounded by sklearn).
**Method:** LogisticRegression, RandomForest, HistGradientBoosting, each fitted on train only and calibrated on the calibration set only. sklearn 1.9 removed `CalibratedClassifierCV(cv='prefit')`, so calibration is explicit: sigmoid = Platt scaling (LogisticRegression on the base model's calibration-set probabilities), isotonic = IsotonicRegression on the same calibration probabilities. ECE is computed by hand with 10 equal-width bins.

## Results — test set (copied from `results/results.txt`)

| Model | Variant | Brier | ROC-AUC | ECE |
|---|---|---|---|---|
| LogisticRegression | uncalibrated | 0.0657 | 0.9397 | 0.0657 |
| LogisticRegression | sigmoid | 0.0740 | 0.9397 | 0.0858 |
| LogisticRegression | isotonic | 0.0732 | 0.9133 | 0.0530 |
| RandomForest | uncalibrated | 0.0691 | 0.9419 | 0.0573 |
| RandomForest | sigmoid | 0.0799 | 0.9419 | 0.0909 |
| RandomForest | isotonic | 0.0688 | 0.9306 | 0.0521 |
| HistGradientBoosting | uncalibrated | 0.0599 | 0.9614 | 0.0452 |
| HistGradientBoosting | sigmoid | 0.0669 | 0.9614 | 0.0902 |
| HistGradientBoosting | isotonic | 0.0671 | 0.9538 | 0.0516 |

**Recommendation:** HistGradientBoosting + sigmoid — lowest calibrated Brier score on test (0.0669). Honest caveat: no calibrated variant beat the best uncalibrated model (HistGradientBoosting uncalibrated Brier 0.0599); with only 98 calibration rows, calibration hurt Brier here.

## Run

```bash
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```

Outputs: `results/results.txt`, `visualizations/reliability_best_worst.png` (reliability diagrams for the best/worst models by uncalibrated Brier).
