"""Lab 18 - Streaming Simulation (UCI Student Performance, student-por.csv).
NO timestamp exists in this dataset, so FILE ORDER is a SIMULATION order only
- it is not a real time series. Batches: 6 x 100 + 49 = 7 batches, in file order.
Preprocessing is fit on batch 0 ONLY; scaled values clipped to [-10, 10].
Target: pass = (G3 >= 10). G3 never a feature.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "shared_dataset" / "dataset.csv", sep=";")
y = (df["G3"] >= 10).astype(int).values; X = df.drop(columns=["G3"]).reset_index(drop=True)
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
batches = [(X[i:i+100], y[i:i+100]) for i in range(0, len(X), 100)]
lines = ["Dataset: UCI Student Performance (student-por.csv), 649 rows",
         "SIMULATION ORDER ONLY: no timestamp column exists; batches are file order, NOT real time.",
         f"Batches of 100 in file order: {len(batches)} batches (6 x 100 + 49); preprocessing fit on batch 0 ONLY; scaled values clipped to [-10, 10]",
         "Target: pass = (G3 >= 10); Pass 549 (84.6%), Fail 100 (15.4%); G3 never a feature"]
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
pre.fit(batches[0][0])
def tr(Xb):
    a = pre.transform(Xb)
    if hasattr(a, "toarray"): a = a.toarray()
    return np.clip(a, -10, 10)
def batch_score(yt, proba):
    if len(np.unique(yt)) < 2: return accuracy_score(yt, (proba >= 0.5).astype(int)), "accuracy (ROC-AUC undefined, single class)"
    return roc_auc_score(yt, proba), "ROC-AUC"
# (a) SGD prequential: batch 0 initialises the model, batches 1.. score-then-update
sgd = SGDClassifier(loss="log_loss", random_state=42)
sgd_scores, sgd_notes = [], []
for i, (Xb, yb) in enumerate(batches):
    Xb_t = tr(Xb)
    if i == 0:
        sgd.partial_fit(Xb_t, yb, classes=np.array([0, 1])); continue
    proba = sgd.predict_proba(Xb_t)[:, 1]
    s, kind = batch_score(yb, proba); sgd_scores.append(s); sgd_notes.append(kind)
    sgd.partial_fit(Xb_t, yb)
    lines.append(f"(a) SGD prequential batch {i}: {kind}={s:.4f} (n={len(yb)})")
mean_sgd = float(np.mean(sgd_scores)) if sgd_scores else float("nan")
lines.append(f"(a) SGD mean prequential score over {len(sgd_scores)} scored batches (batches 1-6; batch 0 used for initial fit): {mean_sgd:.4f}")
# (b) Retraining trigger: RF on all data seen so far; retrain only if last batch ROC-AUC < 0.70
THRESH = 0.70
lines.append(f"(b) Retraining threshold (stated): retrain only when last batch ROC-AUC < {THRESH} (accuracy used if ROC-AUC undefined for a batch)")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
seen_X, seen_y = [batches[0][0]], [batches[0][1]]
rf.fit(tr(batches[0][0]), batches[0][1])
rf_scores, triggers = [], 0
for i, (Xb, yb) in enumerate(batches):
    if i == 0: continue
    proba = rf.predict_proba(tr(Xb))[:, 1]
    s, kind = batch_score(yb, proba); rf_scores.append(s)
    trig = s < THRESH
    lines.append(f"(b) Trigger approach batch {i}: {kind}={s:.4f} (n={len(yb)}) -> {'RETRAIN' if trig else 'keep model'}")
    seen_X.append(Xb); seen_y.append(yb)
    if trig:
        triggers += 1
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(tr(pd.concat(seen_X, ignore_index=True)), np.concatenate(seen_y))
lines.append(f"(b) Number of retraining triggers: {triggers} out of {len(batches)-1} evaluated batches")
lines.append(f"(b) Final-batch (batch {len(batches)-1}) performance: {rf_scores[-1]:.4f}")
(OUT / "results").mkdir(exist_ok=True); (OUT / "visualizations").mkdir(exist_ok=True)
(OUT / "results" / "results.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(batches)), sgd_scores, marker="o", label="SGD prequential")
plt.plot(range(1, len(batches)), rf_scores, marker="s", label="Trigger RF")
plt.axhline(THRESH, linestyle="--", label="Trigger threshold 0.70")
plt.xlabel("Batch index"); plt.ylabel("Score"); plt.legend(); plt.title("Streaming simulation: per-batch scores (file order only)")
plt.tight_layout(); plt.savefig(OUT / "visualizations" / "streaming_scores.png", dpi=150); plt.close()
