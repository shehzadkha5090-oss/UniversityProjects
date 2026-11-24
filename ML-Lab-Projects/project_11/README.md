# Lab 15 — Noise Robustness

**Dataset:** UCI Student Performance (`student-por.csv`), 649 rows, semicolon-delimited, in `../shared_dataset/dataset.csv`.
**Target:** pass = (G3 >= 10). Pass 549 (84.6%), Fail 100 (15.4%). G3 is never a feature.
**Split:** stratified 75/25, seed 42. Actual split produced by scikit-learn: train 486, test 163 (25% of 649 rounds to 163).
**Model:** RandomForest (200 trees) in a Pipeline with median/most-frequent imputation, StandardScaler for numeric features, OneHotEncoder for categorical features, fit on train only.

## Results (from `results/results.txt`, actually run)

- Baseline clean RandomForest: ROC-AUC = 0.9432, F1 = 0.9496, Accuracy = 0.9141
- SIMULATION — flip 10% of train labels: ROC-AUC = 0.9429, F1 = 0.9420, Accuracy = 0.9018
- SIMULATION — flip 20% of train labels: ROC-AUC = 0.8967, F1 = 0.9314, Accuracy = 0.8834
- SIMULATION — Gaussian noise N(0, 0.5) on scaled numeric train features: ROC-AUC = 0.9028, F1 = 0.9366, Accuracy = 0.8896
- SIMULATION — corrupt 10% of train `Mjob` values (random category swaps): ROC-AUC = 0.9432, F1 = 0.9496, Accuracy = 0.9141 — i.e. no measured change on this test set, suggesting `Mjob` carries little weight for this model.
- FIX 1 — OOF-cleaned retrain on 20%-flipped labels (kept 354/486 train rows): ROC-AUC = 0.9422, F1 = 0.9455, Accuracy = 0.9080 — recovered, essentially back to the clean level.
- FIX 2 — regularised RandomForest (max_depth=8) on 20%-flipped labels: ROC-AUC = 0.9342, F1 = 0.9451, Accuracy = 0.9080 — also recovered most of the loss.
- FIX 3 — clip `absences` at the train 95th percentile (12) on the clean split: ROC-AUC = 0.9406, F1 = 0.9534, Accuracy = 0.9202 — no harm, no clear gain.

All noise conditions above are introduced simulations, not original data. Honest conclusion: label noise at 20% and Gaussian feature noise hurt, while both the OOF-cleaning fix and the depth-limited model recovered most of the lost ROC-AUC.

## Run

```bash
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```

## Files

- `main.py` — self-contained experiment
- `results/results.txt` — measured results
- `visualizations/noise_robustness.png` — clean vs label-flip ROC-AUC
