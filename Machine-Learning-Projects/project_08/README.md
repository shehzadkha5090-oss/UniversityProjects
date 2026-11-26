# Lab 10 — Class Imbalance

**Dataset:** UCI Student Performance, `student-por.csv` (649 rows), shared at `../shared_dataset/dataset.csv`.
**Target:** pass = (G3 >= 10); fail is the minority class (label 0).
**Important:** the printed lab assumption (<2% target) is NOT met — the fail rate is 100/649 = 15.4% (pass 549, 84.6%).
**Split:** stratified 75/25, seed 42 — train 486, test 163 (649 × 0.25 rounds up in sklearn, so the test set is 163, not 162). All methods use this same split. SMOTE and undersampling are applied to train only, after preprocessing.

## Results — fail class, test set (copied from `results/results.txt`)

| Method | Precision (fail) | Recall (fail) | F1 (fail) | ROC-AUC |
|---|---|---|---|---|
| Baseline LogisticRegression | 0.7308 | 0.7600 | 0.7451 | 0.9420 |
| class_weight='balanced' | 0.5385 | 0.8400 | 0.6562 | 0.9359 |
| SMOTE (train only) | 0.5882 | 0.8000 | 0.6780 | 0.9368 |
| Random undersampling (train only) | 0.4167 | 0.8000 | 0.5479 | 0.9041 |
| Threshold tuning (thr 0.73, tuned on train OOF, OOF F1_fail 0.7500) | 0.5882 | 0.8000 | 0.6780 | 0.9420 |

**Recommendation:** Baseline — highest test F1 for the fail class (0.7451). The rebalancing methods raised fail recall but cost enough precision to lower F1; at a 15.4% minority rate the gains are modest.

## Run

```bash
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```

Outputs: `results/results.txt`, `visualizations/imbalance_f1.png`.
