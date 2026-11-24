# Lab 14 — Hyperparameter Search

**Dataset:** UCI Student Performance, `student-por.csv` (649 rows), shared at `../shared_dataset/dataset.csv`. RandomForest is run on the full 649 rows — no 15,000-row subsample is claimed, because the dataset has only 649 rows.
**Target:** pass = (G3 >= 10). G3 is never a feature.
**Split:** stratified 75/25, seed 42 — train 486, test 163. Search uses 3-fold CV on train only; the test set is not used for search.
**Bayesian encoding (stated plainly):** max_depth searched on [3, 25] with 25 mapped to None; min_samples_leaf on [1, 10] (integer); n_estimators fixed at 200. Surrogate: GaussianProcessRegressor with a Matérn kernel; acquisition: Expected Improvement. 5 random starts + 7 EI iterations = 12 evaluations, matching the other methods' budget.

## Results — best 3-fold CV ROC-AUC on train (copied from `results/results.txt`)

| Method | Best CV ROC-AUC | Best params | Wall-clock time | Budget |
|---|---|---|---|---|
| Grid search | 0.9702 | n_estimators 100, max_depth 5, min_samples_leaf 2 | 14.5 s | 12 configs |
| Random search | 0.9698 | n_estimators 200, max_depth None, min_samples_leaf 2, max_features 'sqrt' | 14.5 s | 12 configs |
| Manual Bayesian (GP + EI) | 0.9708 | n_estimators 200, max_depth 20, min_samples_leaf 7 | 11.2 s | 12 evaluations |

**Recommendation:** Bayesian search — highest CV ROC-AUC (0.9708) at equal budget. Caveat: the three scores are within 0.001 of each other, so on 486 train rows this is effectively a tie; if simplicity matters, grid search is a defensible pick.

## Run

```bash
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```

Outputs: `results/results.txt`, `visualizations/search_comparison.png`.
