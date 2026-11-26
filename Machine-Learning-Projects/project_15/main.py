"""Lab 19 - Error Analysis (UCI Student Performance, student-por.csv).
5-fold OUT-OF-FOLD RandomForest predictions on the FULL 649 rows (no held-out
test split here, so slices stay large enough). Target: pass = (G3 >= 10).
G3 never a feature. One targeted fix, validated with the same OOF procedure.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "shared_dataset" / "dataset.csv", sep=";")
y = (df["G3"] >= 10).astype(int).values; X = df.drop(columns=["G3"])
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
def pipe():
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    return Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])
def oof_preds(sample_weight_fn=None):
    proba = np.zeros(len(X)); skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for tr_idx, te_idx in skf.split(X, y):
        m = pipe()
        if sample_weight_fn is None:
            m.fit(X.iloc[tr_idx], y[tr_idx])
        else:
            w = sample_weight_fn(tr_idx)
            m.fit(X.iloc[tr_idx], y[tr_idx], rf__sample_weight=w)
        proba[te_idx] = m.predict_proba(X.iloc[te_idx])[:, 1]
    return proba
proba = oof_preds()
pred = (proba >= 0.5).astype(int)
lines = ["Dataset: UCI Student Performance (student-por.csv), FULL 649 rows, 5-fold out-of-fold predictions",
         "Target: pass = (G3 >= 10); Pass 549 (84.6%), Fail 100 (15.4%); G3 never a feature",
         f"Overall OOF: ROC-AUC={roc_auc_score(y, proba):.4f} F1={f1_score(y, pred, zero_division=0):.4f} Accuracy={accuracy_score(y, pred):.4f}"]
def age_band(a):
    return "15-16" if a <= 16 else ("17" if a == 17 else "18+")
slices = {}
slices["sex"] = df["sex"]
slices["school"] = df["school"]
slices["age_band"] = df["age"].map(age_band)
slices["failures"] = np.where(df["failures"] == 0, "0", ">=1")
err = {}
for group, vals in slices.items():
    for val in sorted(pd.Series(vals).unique()):
        mask = (pd.Series(vals).values == val)
        e = 1 - accuracy_score(y[mask], pred[mask])
        err[(group, str(val))] = (e, int(mask.sum()))
        lines.append(f"Error rate by {group}={val}: {e:.4f} (n={int(mask.sum())})")
candidates = {k: v for k, v in err.items() if v[1] >= 30}
worst = max(candidates, key=lambda k: candidates[k][0])
lines.append(f"Worst slice with >=30 rows: {worst[0]}={worst[1]} error rate={candidates[worst][0]:.4f} (n={candidates[worst][1]})")
# ONE targeted fix: upweight (x2) training rows in the worst slice, same OOF procedure
group, val = worst
vals_all = slices[group]
worst_mask_all = (pd.Series(vals_all).values == val)
def weight_fn(tr_idx):
    w = np.ones(len(tr_idx))
    w[worst_mask_all[tr_idx]] = 2.0
    return w
proba2 = oof_preds(weight_fn); pred2 = (proba2 >= 0.5).astype(int)
e2 = 1 - accuracy_score(y[worst_mask_all], pred2[worst_mask_all])
lines.append(f"FIX (one targeted fix): sample weight 2.0 for worst-slice training rows ({group}={val}), same 5-fold OOF procedure")
lines.append(f"FIX result worst slice {group}={val}: error rate {candidates[worst][0]:.4f} -> {e2:.4f} "
             f"({'IMPROVED' if e2 < candidates[worst][0] else 'did NOT improve'})")
lines.append(f"FIX result overall OOF: ROC-AUC={roc_auc_score(y, proba2):.4f} F1={f1_score(y, pred2, zero_division=0):.4f} "
             f"Accuracy={accuracy_score(y, pred2):.4f}")
(OUT / "results").mkdir(exist_ok=True); (OUT / "visualizations").mkdir(exist_ok=True)
(OUT / "results" / "results.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
keys = [f"{g}={v}" for (g, v), _ in sorted(err.items())]
vals = [err[k][0] for k in sorted(err)]
plt.figure(figsize=(10, 4)); plt.bar(keys, vals); plt.ylabel("OOF error rate"); plt.xticks(rotation=30, ha="right")
plt.title("Error analysis: OOF error rate by slice"); plt.tight_layout()
plt.savefig(OUT / "visualizations" / "error_by_slice.png", dpi=150); plt.close()
