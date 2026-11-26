# Lab 18 — Streaming Simulation

**Dataset:** UCI Student Performance (`student-por.csv`), 649 rows, in `../shared_dataset/dataset.csv`.
**Target:** pass = (G3 >= 10). Pass 549 (84.6%), Fail 100 (15.4%). G3 is never a feature.

## Important: simulation order only

There is NO timestamp in this dataset, so file order is a SIMULATION order only — it is not a real time series and the per-batch scores below must not be read as performance over time.

**Setup:** 7 batches in file order (6 × 100 + 49). Preprocessing fit on batch 0 only; scaled values clipped to [-10, 10].

## Results (from `results/results.txt`, actually run)

(a) Incremental SGDClassifier (log_loss, partial_fit), prequential (predict batch, then update on it). Batch 0 was used for the initial fit, so scores cover batches 1–6:

- Batch 1: 0.5000, Batch 2: 0.8498, Batch 3: 0.9966, Batch 4: 0.7619, Batch 5: 0.7821, Batch 6: 0.8868
- Mean prequential ROC-AUC: **0.7962** over the 6 scored batches. The batch-1 score of 0.5000 reflects a model that had only seen 100 rows.

(b) Retraining-trigger approach (RandomForest trained on all data seen so far; retrain only when the last batch's ROC-AUC drops below the stated threshold of 0.70):

- Batch 1: 0.9122 (keep), Batch 2: 0.9459 (keep), Batch 3: 0.9931 (keep), Batch 4: 0.8431 (keep), Batch 5: 0.8689 (keep), Batch 6: 0.8803 (keep)
- Number of retraining triggers: **0 out of 6** evaluated batches
- Final-batch (batch 6) performance: **0.8803**

Honest note: no trigger ever fired, so approach (b) is effectively just the batch-0 model evaluated on later batches — the trigger mechanism was implemented and tested, but the threshold was never crossed on these simulated batches.

## Run

```bash
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```

## Files

- `main.py` — self-contained experiment
- `results/results.txt` — measured results
- `visualizations/streaming_scores.png` — per-batch scores
