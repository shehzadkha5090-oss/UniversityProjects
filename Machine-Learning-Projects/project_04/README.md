# Project 04 — Lab 4: Explainability vs Performance

## Dataset
UCI Student Performance (Portuguese course, `student-por.csv`), shared file
`shared_dataset/dataset.csv`, semicolon-delimited, 649 rows × 33 columns.

## Target and split
Target: pass = (G3 >= 10). Pass 549 (84.6%), fail 100 (15.4%).
Split: stratified 75/25, seed 42 — train 486, test 163.

## Method
Same split for three models: LogisticRegression (interpretable),
DecisionTree with max_depth=4 (interpretable), RandomForest with 200 trees
(accurate but opaque). The forest is explained with permutation importance
on the test set and SHAP (TreeExplainer, first 100 test rows — stated
subsample, used because full-test SHAP is slow).

## Results (from `results/results.txt`, test set of 163)
| Model | ROC-AUC | Accuracy |
|---|---|---|
| LogisticRegression | 0.9420 | 0.9202 |
| DecisionTree (max_depth=4) | 0.7797 | 0.8896 |
| RandomForest (200 trees) | 0.9432 | 0.9141 |

Top permutation importance (test ROC-AUC drop): G2 0.1262, G1 0.0644,
famrel 0.0111, then freetime/reason/health at ~0.002. Top mean |SHAP value|
(first 100 test rows): G1 0.0793, G2 0.0744, school_GP 0.0197,
failures 0.0191, school_MS 0.0173. Both explanations agree that earlier
grades dominate.

## Trade-off discussion
On this dataset there is almost no performance price for interpretability
at the top end: LogisticRegression (0.9420) essentially matches the forest
(0.9432) while its coefficients can be read directly, and its accuracy
(0.9202) is even slightly higher. The depth-4 tree is genuinely weaker
(0.7797 ROC-AUC) — the price of a model simple enough to draw. The forest
buys only a negligible ROC-AUC gain over logistic regression here, at the
cost of needing post-hoc explanation (permutation + SHAP) rather than being
interpretable by construction. For a school-facing early-warning tool, the
logistic model is the better trade-off on these numbers.

## How to run
```
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```
Outputs: `results/results.txt`,
`visualizations/permutation_importance.png`, `visualizations/shap_summary.png`.
