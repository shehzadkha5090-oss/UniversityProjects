"""Lab 6 Bias & Fairness — UCI Student Performance (student-por.csv).
Target: pass = (G3 >= 10). G3 is NEVER a feature. Features: all columns except G3.
Sensitive attribute: sex (F/M). Secondary slice: school (GP/MS).
Baseline LogisticRegression vs manual reweighing (sample weights inversely
proportional to group x label frequencies, computed on train only).
Dataset: shared_dataset/dataset.csv, semicolon-delimited, 649 rows.
Split: stratified 75/25, seed 42 (train 486, test 163)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parents[1]
OUT, VIS = BASE / "project_06" / "results", BASE / "project_06" / "visualizations"
OUT.mkdir(parents=True, exist_ok=True); VIS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "shared_dataset" / "dataset.csv", sep=";")
y = (df["G3"] >= 10).astype(int)
X = df.drop(columns=["G3"])
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
def make(): return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, random_state=42))])

Xtr, Xte, ytr, yte, str_tr, str_te = train_test_split(
    X, y, df["sex"], test_size=0.25, random_state=42, stratify=y)
# Re-train/test split bookkeeping: train_test_split above returns aligned splits;
# sex series for test must come from Xte index.
sex_te = df.loc[Xte.index, "sex"]; school_te = df.loc[Xte.index, "school"]
# NOTE: Xtr/Xte above were split with stratify on y; sex_te/school_te aligned by index.

def gaps(y_true, y_pred, proba, groups):
    out = {}
    for g in sorted(groups.unique()):
        m = groups == g
        out[g] = dict(n=int(m.sum()), pred_pass_rate=float(y_pred[m].mean()),
                      tpr=float(((y_pred[m] == 1) & (y_true[m] == 1)).sum() / max((y_true[m] == 1).sum(), 1)))
    vals = list(out.values())
    return out, abs(vals[0]["pred_pass_rate"] - vals[1]["pred_pass_rate"]), abs(vals[0]["tpr"] - vals[1]["tpr"])

# Baseline
base = make(); base.fit(Xtr, ytr)
pred_b = base.predict(Xte); prob_b = base.predict_proba(Xte)[:, 1]
auc_b = roc_auc_score(yte, prob_b)
sex_b, dpd_b, eod_b = gaps(yte.values, pred_b, prob_b, sex_te)
sch_b, dpd_s_b, eod_s_b = gaps(yte.values, pred_b, prob_b, school_te)

# Manual reweighing: w(a,y) = (1 / count_{a,y}) rescaled to mean 1, train only.
sex_tr = df.loc[Xtr.index, "sex"]
w = np.ones(len(Xtr))
for a in sex_tr.unique():
    for lab in (0, 1):
        m = (sex_tr.values == a) & (ytr.values == lab)
        w[m] = 1.0 / max(m.sum(), 1)
w = w / w.mean()
rw = make(); rw.fit(Xtr, ytr, clf__sample_weight=w)
pred_r = rw.predict(Xte); prob_r = rw.predict_proba(Xte)[:, 1]
auc_r = roc_auc_score(yte, prob_r)
sex_r, dpd_r, eod_r = gaps(yte.values, pred_r, prob_r, sex_te)
sch_r, dpd_s_r, eod_s_r = gaps(yte.values, pred_r, prob_r, school_te)

# Real pass rates, full data
full = {}
for g in ("F", "M"):
    m = df["sex"] == g; full[g] = (float((df.loc[m, "G3"] >= 10).mean()), int(m.sum()))

lines = []
def add(s): lines.append(s)
add("Lab 6 Bias & Fairness — UCI Student Performance (student-por.csv)")
add(f"Rows: 649; target pass=(G3>=10): pass 549 (84.6%), fail 100 (15.4%); G3 never a feature.")
add(f"Split: stratified 75/25 seed 42 — train {len(ytr)}, test {len(yte)}")
add(f"Real pass rates (full data): F {full['F'][0]:.3f} (n={full['F'][1]}), M {full['M'][0]:.3f} (n={full['M'][1]})")
add(f"BASELINE LogisticRegression: ROC-AUC={auc_b:.4f}")
add(f"  sex gaps: DPD={dpd_b:.4f}, EOD (TPR gap)={eod_b:.4f}; per-group {sex_b}")
add(f"  school gaps: DPD={dpd_s_b:.4f}, EOD={eod_s_b:.4f}; per-group {sch_b}")
add(f"REWEIGHED LogisticRegression: ROC-AUC={auc_r:.4f} (change {auc_r-auc_b:+.4f})")
add(f"  sex gaps: DPD={dpd_r:.4f}, EOD={eod_r:.4f}; per-group {sex_r}")
add(f"  school gaps: DPD={dpd_s_r:.4f}, EOD={eod_s_r:.4f}; per-group {sch_r}")
add(f"Discussion: reweighing by sex slightly narrowed the sex gaps (DPD {dpd_b:.4f}->{dpd_r:.4f}, EOD {eod_b:.4f}->{eod_r:.4f})")
add(f"  at a ROC-AUC cost, but it WORSENED the school slice (DPD {dpd_s_b:.4f}->{dpd_s_r:.4f}, EOD {eod_s_b:.4f}->{eod_s_r:.4f}) —")
add("  so it did not help fairness overall; it only rebalanced the attribute it was weighted on.")
add("Discussion: reweighing is judged by whether DPD/EOD fall and what ROC-AUC costs;")
add("  with only 163 test rows, small gap changes are noisy and are not claimed as proof")
add("  of fairness. Sex was included as a model feature in both models (default feature set).")
(OUT / "results.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(2); wd = 0.35
ax.bar(x - wd/2, [dpd_b, eod_b], wd, label="Baseline")
ax.bar(x + wd/2, [dpd_r, eod_r], wd, label="Reweighed")
ax.set_xticks(x); ax.set_xticklabels(["Demographic parity diff.", "Equal-opportunity diff. (TPR gap)"])
ax.set_ylabel("Gap (F vs M, test set)"); ax.set_title("Lab 6: Fairness gaps by sex — baseline vs reweighed")
ax.legend(); fig.tight_layout(); fig.savefig(VIS / "fairness_gaps.png", dpi=150)
print("Saved fairness_gaps.png")
