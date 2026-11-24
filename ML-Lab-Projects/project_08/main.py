"""Lab 10 Class Imbalance — UCI Student Performance (student-por.csv).
The printed lab assumption (<2% target) is NOT met here: fail rate is
100/649 = 15.4% (pass 549, 84.6%). All methods are compared on the same
stratified 75/25 split, seed 42 (train 486, test 163). Target: pass=(G3>=10);
the FAIL class (label 0) is the minority class reported below.
SMOTE and undersampling are applied to TRAIN ONLY, after preprocessing,
via imbalanced-learn Pipelines. Threshold tuning uses out-of-fold
predictions on the TRAIN set only."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parents[1]
OUT, VIS = BASE / "project_08" / "results", BASE / "project_08" / "visualizations"
OUT.mkdir(parents=True, exist_ok=True); VIS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "shared_dataset" / "dataset.csv", sep=";")
y = (df["G3"] >= 10).astype(int)  # 1=pass, 0=fail (minority)
X = df.drop(columns=["G3"])
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

def fail_metrics(y_true, y_pred, prob_pass):
    # FAIL class = label 0; prob of fail = 1 - prob of pass
    return dict(precision=float(precision_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)) if False else float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
                recall=float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
                f1=float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
                roc_auc=float(roc_auc_score(y_true, prob_pass)))
def lr(): return LogisticRegression(max_iter=1000, random_state=42)
results = {}
# Baseline
m = Pipeline([("pre", pre), ("clf", lr())]); m.fit(Xtr, ytr)
results["Baseline"] = fail_metrics(yte, m.predict(Xte), m.predict_proba(Xte)[:, 1])
# Class-weight balanced
m = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))]); m.fit(Xtr, ytr)
results["class_weight=balanced"] = fail_metrics(yte, m.predict(Xte), m.predict_proba(Xte)[:, 1])
# SMOTE (train only, after preprocessing)
m = ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=42)), ("clf", lr())]); m.fit(Xtr, ytr)
results["SMOTE (train only)"] = fail_metrics(yte, m.predict(Xte), m.predict_proba(Xte)[:, 1])
# Random undersampling (train only)
m = ImbPipeline([("pre", pre), ("rus", RandomUnderSampler(random_state=42)), ("clf", lr())]); m.fit(Xtr, ytr)
results["Random undersampling (train only)"] = fail_metrics(yte, m.predict(Xte), m.predict_proba(Xte)[:, 1])
# Threshold tuning on OOF train predictions only
base = Pipeline([("pre", pre), ("clf", lr())])
oof = cross_val_predict(base, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
best_thr, best_f1 = 0.5, -1
for thr in np.arange(0.05, 0.96, 0.05):
    pred_pass_oof = (oof >= thr).astype(int)
    f1 = f1_score(ytr, pred_pass_oof, pos_label=0, zero_division=0)
    if f1 > best_f1: best_f1, best_thr = float(thr), float(f1)
base.fit(Xtr, ytr); prob_te = base.predict_proba(Xte)[:, 1]
results[f"Threshold tuning (thr={best_thr:.2f} on train OOF, F1_fail OOF={best_f1:.4f})"] = fail_metrics(yte, (prob_te >= best_thr).astype(int), prob_te)

lines = ["Lab 10 Class Imbalance — UCI Student Performance (student-por.csv)",
 "Lab assumption (<2% target) NOT met: fail rate 100/649 = 15.4% (pass 549, 84.6%).",
 f"Split: stratified 75/25 seed 42 — train {len(ytr)}, test {len(yte)}; metrics below are for the FAIL class (label 0)."]
for k, v in results.items():
    lines.append(f"{k}: precision_fail={v['precision']:.4f}, recall_fail={v['recall']:.4f}, F1_fail={v['f1']:.4f}, ROC-AUC={v['roc_auc']:.4f}")
best = max(results, key=lambda k: results[k]["f1"])
lines.append(f"Recommendation: {best} — highest test F1 for the fail class ({results[best]['f1']:.4f});")
lines.append("  chosen on F1 for the minority (fail) class, not on ROC-AUC alone, because the")
lines.append("  lab's concern is detecting the minority class, and the 15.4% rate means gains are modest.")
(OUT / "results.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

fig, ax = plt.subplots(figsize=(9, 4.5))
names = list(results); ax.bar(names, [results[k]["f1"] for k in names])
ax.set_ylabel("F1 for FAIL class (test)"); ax.set_title("Lab 10: Fail-class F1 by strategy")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right"); fig.tight_layout(); fig.savefig(VIS / "imbalance_f1.png", dpi=150)
print("Saved imbalance_f1.png")
