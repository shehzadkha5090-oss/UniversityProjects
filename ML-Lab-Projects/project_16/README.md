# Lab 7 — High-Dimensional Feature Selection (project_16)

## Caveat — read first

**CAVEAT: The original dataset has only 32 features, NOT 10,000+, so the lab's
stated condition is NOT met. This project is an honest adaptation, not a claim
that the condition holds.**

## Dataset and target

- Dataset: UCI Student Performance, `student-por.csv` ONLY
  (`../shared_dataset/dataset.csv`, 649 rows × 33 cols, semicolon-delimited, read with `sep=';'`).
  `student-mat.csv` was NOT used.
- Target: pass = (G3 >= 10); pass 549 (84.6%), fail 100 (15.4%).
- G3 was NEVER used as a feature. Default features: all columns except G3 (32 features).
- Split: stratified 75/25 train/test, seed 42 (train 486, test 163).
- Preprocessing (Pipeline, fit on train only): numeric median-impute + StandardScaler;
  categorical most-frequent-impute + OneHotEncoder(handle_unknown='ignore').

## Adaptation: how p was reached

Standard preprocessing yields 58 features (15 numeric + 43 one-hot). A second-degree
polynomial expansion (`PolynomialFeatures`, degree=2, `include_bias=False`) was then
applied to ALL 58 preprocessed features, so:

**p = 58 + 58×59/2 = 1769 features.**

p > n (1769 > 649 rows in total, and > 486 training rows), but p is NOT 10,000, so
this project does NOT claim 10,000 features.

## Results (same split for every method)

| Method | Features used | Train ROC-AUC | Test ROC-AUC | Test accuracy |
|---|---:|---:|---:|---:|
| Reference: L2 LR on 58 preprocessed features (no expansion) | 58 | 0.9859 | 0.9420 | 0.9202 |
| (a) L2 LR, all expanded features | 1769 | 1.0000 | 0.9377 | 0.9202 |
| (b) L1 LR (saga) selection | 22 | 0.9854 | 0.9664 | 0.9264 |
| (c) SelectKBest f_classif k=50 + L2 LR | 50 | 0.9831 | 0.9649 | 0.9202 |
| (d) Stability selection (>=50%) + L2 LR | 13 | 0.9868 | 0.9577 | 0.8957 |

Details:

- (b) L1: `penalty='l1'`, `solver='saga'`, C=0.1, `max_iter=1000`; 22 non-zero coefficients out of 1769.
- (c) SelectKBest: `f_classif`, k=50.
- (d) Stability selection: L1 (saga, C=0.1, `max_iter=1000`) refit on 20 bootstrap resamples of
  the training set; features kept if selected in >= 50% of resamples (>= 10 of 20);
  13 features kept, then L2 LR on those features.

## Overlap: L1 vs stability selection

- L1-selected: 22, stability-selected: 13, in common: 13, union: 22, **Jaccard = 0.5909**.
- All 13 stability-selected features are also L1-selected; the Jaccard is below 1.0 only
  because the L1 set is larger (22 vs 13).

## Discussion — p>>n overfitting, honestly

Method (a), all 1769 features: train ROC-AUC = 1.0000 but test ROC-AUC = 0.9377 —
a train-test gap of 0.0623. With p (1769) far above the 486 training rows, the L2 model
can essentially memorise the training set while generalising worse, which is exactly the
p>>n failure mode. Selection methods (b), (c) and (d) use far fewer features, do not show
the same train-test gap, and beat (a) on test ROC-AUC (0.9664 / 0.9649 / 0.9577 vs 0.9377)
— though stability selection's test accuracy (0.8957) is below (a)'s (0.9202), so the gain
is in ranking, not in every metric. Selection does not automatically beat the simpler
58-feature reference model (test ROC-AUC 0.9420). The honest reading is that expansion to
p=1769 bought no reliable gain over the 58-feature model on this small dataset; selection
mainly recovered comparable-or-better test ROC-AUC with far fewer features, rather than
transforming performance. No result here should be read as evidence about true 10,000+
feature settings, because that condition is not met (see Caveat above).

## Files

- `main.py` — self-contained script; loads the dataset via
  `Path(__file__).resolve().parents[1] / 'shared_dataset' / 'dataset.csv'`.
  Run with `~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py`.
- `results/results.txt` — full results, selected feature lists, and discussion.
- `visualizations/roc_auc_by_method.png` — test ROC-AUC by method, and features used
  (log scale) vs test ROC-AUC.
