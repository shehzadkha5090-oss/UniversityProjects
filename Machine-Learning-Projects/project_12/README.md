# Lab 16 — Ensemble Diversity

**Dataset:** UCI Student Performance (`student-por.csv`), 649 rows, in `../shared_dataset/dataset.csv`.
**Target:** pass = (G3 >= 10). Pass 549 (84.6%), Fail 100 (15.4%). G3 is never a feature.
**Split:** stratified 75/25, seed 42. Actual split: train 486, test 163.
**Base models:** LogisticRegression, RandomForest, HistGradientBoosting, KNN — each in its own Pipeline (median/most-frequent impute, StandardScaler, OneHotEncoder), same split.

## Results (from `results/results.txt`, actually run)

Single models, test ROC-AUC:

- LogisticRegression: 0.9420
- RandomForest: 0.9432
- HistGradientBoosting: 0.9597
- KNN: 0.7996

- Soft-voting ensemble (all 4): ROC-AUC = 0.9403
- Diversity (pairwise): the most redundant pair was RandomForest vs HistGradientBoosting (disagreement rate 0.0184, Q-statistic 0.9979); the most diverse pairs involved KNN (disagreement ~0.09–0.10, Q ~0.92–0.94).
- Mean Q per model: LogisticRegression 0.9665, RandomForest 0.9733, HistGradientBoosting 0.9703, KNN 0.9289.
- Pruned model (highest mean Q, most redundant): RandomForest.
- Pruned soft-voting ensemble (without RandomForest): ROC-AUC = 0.9368.

**Honest discussion:** the best single model won. HistGradientBoosting (0.9597) beat both the full voting ensemble (0.9403) and the pruned ensemble (0.9368). Diversity did not help on this test set — KNN was diverse but much weaker, and it dragged the vote down, while the three strong models agreed with each other most of the time.

## Run

```bash
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```

## Files

- `main.py` — self-contained experiment (pairwise disagreement and Q-statistic computed by its own code)
- `results/results.txt` — measured results
- `visualizations/ensemble_diversity.png` — single models vs ensembles
