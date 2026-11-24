# Lab 8 — Anomaly Detection

**Dataset:** UCI Student Performance, `student-por.csv` (649 rows), shared at `../shared_dataset/dataset.csv`.
**Method:** Unsupervised only — no labels were used for training. IsolationForest and LocalOutlierFactor, each on the full 649 rows (no subsample needed on data this small), contamination = 0.10. Features: all columns except G3.

## Results (copied from `results/results.txt`)

- IsolationForest flagged: 65 rows. LocalOutlierFactor flagged: 65 rows.
- Both flagged: 35; union: 95; Jaccard overlap: 0.3684.
- Proxy-only validation (no ground-truth anomalies exist, so this is NOT accuracy):
  - IsolationForest flagged: fail rate 0.400, mean G3 9.45, mean absences 5.85.
  - LOF flagged: fail rate 0.369, mean G3 10.05, mean absences 6.40.
  - Flagged by both: fail rate 0.486, mean G3 9.00, mean absences 6.20.
  - Unflagged by either: fail rate 0.121, mean G3 12.23, mean absences 3.24.

Flagged rows look worse on these proxies, but that does not prove they are true anomalies.

## Run

```bash
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```

Outputs: `results/results.txt`, `visualizations/anomaly_pca_scatter.png`.
