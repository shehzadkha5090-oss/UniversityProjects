# Lab 20 — End-to-End System Design

**Dataset:** UCI Student Performance, Portuguese class (`student-por.csv`),
shared at `~/workspace/ML_Lab_Projects_v4/shared_dataset/dataset.csv`,
semicolon-delimited, 649 rows × 33 columns. No other dataset was used
(`student-mat.csv` was not used).

**Target:** pass = G3 ≥ 10. **G3 is never used as a feature** — it defines
the target only.

## Actual results (copied from `results/results.txt`, run 2026-10-29)

- Validation: rows = 649, missing = 0, G3 range = 0–19 (allowed 0–20),
  pass rate (G3 ≥ 10) = 0.8459 (84.59%)
- Split: stratified 75/25, seed 42, train = 486, test = 163
  (train pass rate 0.8457, test pass rate 0.8466)
- **Test ROC-AUC = 0.9597**
- **Test accuracy = 0.9080**
- **Test fail-class F1 (label 0) = 0.6809**
- Saved model: `results/model.joblib` (235,922 bytes), reload-verified
  (loaded back in the same run and predicted on 5 test rows)
- Example predictions (first 3 test rows):
  - Test row 0: P(pass) = 1.0000, predicted = pass, true = pass
  - Test row 1: P(pass) = 1.0000, predicted = pass, true = pass
  - Test row 2: P(pass) = 1.0000, predicted = pass, true = pass
- Drift (KS, train vs test): G1 = 0.0632, G2 = 0.0404, absences = 0.0434,
  mean KS = 0.0490 — verdict: **NO RETRAINING TRIGGER** (see below)
- Architecture diagram: `visualizations/architecture.png` (69,470 bytes)

## Chosen model and justification

**Chosen: HistGradientBoostingClassifier**, inside a single scikit-learn
`Pipeline` with the preprocessing below, re-fit from scratch in `main.py`
(no fitted model was copied from an earlier project).

Justification, with numbers from earlier projects on the same split protocol:

- Project 12 (model comparison): HistGradientBoosting ROC-AUC = **0.9597**,
  the best single model — ahead of RandomForest (0.9432),
  LogisticRegression (0.9420) and KNN (0.7996); the soft-voting ensemble
  of all four scored only 0.9403, so the ensemble did not beat it either.
- Project 4: LogisticRegression ROC-AUC = 0.9420, RandomForest = 0.9432 —
  again below HistGradientBoosting.
- Project 9: HistGradientBoosting had the lowest uncalibrated Brier score
  (0.0599) among the models compared there.

The re-fit here reproduces exactly that headline figure on the held-out
test set: **ROC-AUC 0.9597**. LogisticRegression would have been the
simpler, more interpretable fallback, but on this data it costs about
0.018 ROC-AUC, so HistGradientBoosting was chosen. Note the trade-off:
fail-class F1 here is 0.6809 — the minority (fail, ~15%) class is harder
than overall accuracy (0.9080) suggests, so the fail class is monitored
explicitly (see Monitoring).

## Preprocessing (fit on train only)

- Numeric (15: age, Medu, Fedu, traveltime, studytime, failures, famrel,
  freetime, goout, Dalc, Walc, health, absences, G1, G2):
  median imputation + `StandardScaler`.
- Categorical (17: school, sex, address, famsize, Pstatus, Mjob, Fjob,
  reason, guardian, schoolsup, famsup, paid, activities, nursery, higher,
  internet, romantic): most-frequent imputation +
  `OneHotEncoder(handle_unknown='ignore')`.
- Everything is wrapped in one `Pipeline` (`preprocess` → classifier) so
  the exact same transformations apply at serving time and no information
  from the test set leaks into fitting.

## System design — architecture

```
ingest -> validate -> preprocess -> train -> evaluate -> serve -> monitor -> retrain
                                                              (loop back to train)
```

See `visualizations/architecture.png` for the diagram.

1. **Ingest** — read the shared CSV with `sep=';'` via
   `Path(__file__).resolve().parents[1] / 'shared_dataset' / 'dataset.csv'`.
   The loader is deliberately tied to that shared path, not to a local copy.
2. **Validate** — hard asserts, failing loudly (script exits) if any fail:
   649 rows, 33 columns, 0 missing values, G3 within 0–20, pass rate
   within 0.84–0.85 (observed 0.8459, about 84.6%). A failed check means
   the pipeline stops before any training, so a corrupted or swapped
   dataset can never silently produce a model.
3. **Preprocess** — as above, fit on train only.
4. **Train** — HistGradientBoostingClassifier, stratified 75/25 split,
   seed 42 (train 486, test 163).
5. **Evaluate** — ROC-AUC, accuracy and fail-class F1 on the held-out
   test set only (0.9597 / 0.9080 / 0.6809).
6. **Serve** — the trained pipeline is saved to `results/model.joblib`
   with `joblib` and immediately loaded back and used to predict on
   5 test rows to prove the artefact works.
7. **Monitor** — see Monitoring below.
8. **Retrain** — triggered only by the rule below; a retrain re-runs
   `main.py` end to end (ingest → validate → … → new artefact), so the
   retrained model is always built from validated data with the same
   preprocessing.

## Deployment notes

No real server is required for this lab. The saved artefact is a single
self-contained pipeline (preprocessing + classifier), so serving it is a
plain prediction function — the same object that was verified on reload:

```python
import joblib, pandas as pd

model = joblib.load("results/model.joblib")  # preprocessing included

def predict_pass(student_row: dict) -> dict:
    """student_row: one student's features, WITHOUT G3.
    Returns predicted label and P(pass)."""
    X = pd.DataFrame([student_row])
    proba = float(model.predict_proba(X)[0, 1])
    return {"predicted": "pass" if proba >= 0.5 else "fail",
            "P(pass)": round(proba, 4)}
```

To serve it for real, wrap that function in a minimal API (e.g. FastAPI
`POST /predict` accepting the 32 feature fields, rejecting any request
that includes G3 or is missing a required column), run it behind a
versioned artefact store (keep each `model.joblib` with its
`results.txt`), and log every prediction's inputs, probability and
timestamp for monitoring. The `handle_unknown='ignore'` encoder means
unseen categories do not crash serving — they are encoded as all-zeros
for that feature — but they should still be logged as a data-quality
signal (see Failure modes).

## Monitoring metrics

Checked on every new batch of scored students, and compared against the
held-out baseline (ROC-AUC 0.9597, accuracy 0.9080, fail F1 0.6809):

- **ROC-AUC** on newly labelled data (ranking quality).
- **Fail-class F1** — the minority class that matters for early-warning
  use; a drop here matters more than a small accuracy move.
- **Pass-rate drift** — the scored batch's predicted/observed pass rate
  vs the training baseline of 84.6%; a sustained shift suggests the
  population or the grading context has changed.
- **Input distribution drift** — two-sample KS tests on key numeric
  features (G1, G2, absences, extendable to the rest) between the
  training set and the new batch.

## Retraining trigger

**Rule (stated in `main.py` before the verdict): retraining fires if
any KS statistic > 0.20, or the mean KS across the checked features
> 0.10.**

Observed on the held-out test set: G1 = 0.0632, G2 = 0.0404,
absences = 0.0434, mean KS = 0.0490. No feature exceeds 0.20 and the
mean (0.0490) is below 0.10, so the verdict is **NO RETRAINING TRIGGER**.
A retrain would also be warranted outside this rule if ROC-AUC or
fail-class F1 fell materially on newly labelled data, or if the
pass rate moved persistently away from 84.6%.

## Failure modes

- **Missing columns** — a serving request or a new CSV missing a
  required feature fails at the pipeline (and at the validation asserts
  for training data). Fix: reject the batch, log the missing columns,
  do not silently fill with defaults beyond the trained imputers.
- **New categories** — unseen categorical values (e.g. a new job type)
  are handled by `handle_unknown='ignore'` (no crash), but the model
  has no signal for them; a spike in unseen categories should trigger a
  data review and, if persistent, a retrain including the new values.
- **Distribution shift between schools** — earlier work (project 3)
  showed cross-school transfer is fragile (e.g. training on one school
  and testing on the other dropped ROC-AUC sharply for restricted
  feature sets). A new school, or a changed mix of schools, is the most
  likely real-world drift source: monitor per-school metrics, not just
  the global average, before trusting predictions on a new school.
- **Small-sample noise** — with only 649 rows (163 in test, ~25 fails),
  fail-class F1 and KS statistics are noisy; a single batch's dip is
  not proof of drift. Confirm a trigger on a second batch before
  retraining, and keep the previous artefact so a retrain can be rolled
  back if it scores worse on the held-out set.
- **Leakage** — G3 (and anything derived from it) must never enter the
  features; project 1 showed including G3 inflates ROC-AUC to 1.0000.
  The pipeline drops G3 before splitting and asserts it is absent.
- **Stale artefact / version skew** — serving a `model.joblib` whose
  preprocessing does not match the current feature list. Mitigation:
  the pipeline is saved as one object, and each artefact ships with its
  `results.txt` (metrics, split, validation) for traceability.

## How to run

```bash
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```

Outputs: `results/results.txt`, `results/model.joblib`,
`visualizations/architecture.png` — all verified non-empty by the run
itself (asserts on file size and on reload-prediction).
