# Lab 19 — Error Analysis

**Dataset:** UCI Student Performance (`student-por.csv`), FULL 649 rows, in `../shared_dataset/dataset.csv`.
**Target:** pass = (G3 >= 10). Pass 549 (84.6%), Fail 100 (15.4%). G3 is never a feature.
**Method:** 5-fold out-of-fold RandomForest predictions on the full 649 rows (no separate held-out test set here, so slices stay large enough).

## Results (from `results/results.txt`, actually run)

- Overall OOF: ROC-AUC = 0.9669, F1 = 0.9564, Accuracy = 0.9245

Error rate by slice:

- sex = F: 0.0548 (n=383); sex = M: 0.1053 (n=266)
- school = GP: 0.0449 (n=423); school = MS: 0.1327 (n=226)
- age band 15–16: 0.0588 (n=289); 17: 0.0950 (n=179); 18+: 0.0829 (n=181)
- failures = 0: 0.0510 (n=549); failures >= 1: 0.2100 (n=100)

**Worst slice with at least 30 rows: failures >= 1, error rate 0.2100 (n=100).**

**One targeted fix:** sample weight 2.0 for worst-slice training rows (failures >= 1), validated with the same 5-fold OOF procedure.

- Worst-slice error rate: 0.2100 → 0.2000 — **improved, but only slightly** (one fewer error out of 100).
- Overall OOF after the fix: ROC-AUC = 0.9695, F1 = 0.9564, Accuracy = 0.9245 — essentially unchanged overall, so the fix helped the target slice a little without harming overall performance.

## Run

```bash
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```

## Files

- `main.py` — self-contained experiment
- `results/results.txt` — measured results
- `visualizations/error_by_slice.png` — OOF error rate by slice
