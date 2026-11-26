# Lab 17 — Dimensionality Reduction

**Dataset:** UCI Student Performance (`student-por.csv`), FULL 649 rows (no train/test split here), in `../shared_dataset/dataset.csv`.
**Target:** pass = (G3 >= 10). Pass 549 (84.6%), Fail 100 (15.4%). G3 is never an input feature; it is used only for the descriptive silhouette score and for colouring the scatter plots.
**Preprocessing:** median/most-frequent imputation, StandardScaler, OneHotEncoder → 58 features.

## Pre-stated recommendation rule (fixed before seeing results)

Recommend the method with the highest 5-NN cross-validated accuracy in the 2-D space; ties broken by higher trustworthiness. This rule is also written in `main.py` before any result is computed, and it is applied as written below.

## Results (from `results/results.txt`, actually run)

| Method | Silhouette (descriptive only) | 5-NN CV accuracy | Trustworthiness |
|---|---|---|---|
| PCA (2 components) | 0.2380 | 0.8536 | 0.7578 |
| t-SNE (2 components, perplexity 30) | 0.1511 | 0.8458 | 0.9251 |
| UMAP (2 components, n_neighbors 15) | 0.1493 | 0.8551 | 0.8470 |

**Recommended by the pre-stated rule: UMAP** (5-NN CV accuracy 0.8551, trustworthiness 0.8470). Note the margins are small: all three methods keep 5-NN accuracy near the majority-class level, and the silhouette scores are descriptive only — they do not prove the classes separate cleanly in 2-D.

## Run

```bash
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```

## Files

- `main.py` — self-contained experiment
- `results/results.txt` — measured results
- `visualizations/pca_scatter.png`, `tsne_scatter.png`, `umap_scatter.png` — 2-D scatter plots coloured by pass/fail
