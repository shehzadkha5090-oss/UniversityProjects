"""Lab 15 - Noise Robustness (UCI Student Performance, student-por.csv).
Target: pass = (G3 >= 10). G3 is NEVER a feature. All noise conditions below
are LABELLED SIMULATIONS introduced by this script - they are NOT in the
original dataset. Original data: 649 rows, semicolon-delimited.
Split: stratified 75/25, seed 42 (train 487, test 162).
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
rng = np.random.RandomState(42)

df = pd.read_csv(BASE / "shared_dataset" / "dataset.csv", sep=";")
y = (df["G3"] >= 10).astype(int)
X = df.drop(columns=["G3"])
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()

def make_pipe(rf_kwargs=None):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                **(rf_kwargs or {}))
    return Pipeline([("pre", pre), ("rf", rf)])

def scores(model, Xt, yt):
    p = model.predict_proba(Xt)[:, 1]; pr = (p >= 0.5).astype(int)
    return roc_auc_score(yt, p), f1_score(yt, pr, zero_division=0), accuracy_score(yt, pr)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
ytr = ytr.reset_index(drop=True); yte = yte.reset_index(drop=True)
Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
lines = [f"Dataset: UCI Student Performance (student-por.csv), 649 rows",
         f"Target: pass = (G3 >= 10); Pass {(y==1).sum()} ({(y==1).mean()*100:.1f}%), Fail {(y==0).sum()}",
         f"Split: stratified 75/25, seed 42; train {len(Xtr)}, test {len(Xte)} (G3 never a feature)",
         "ALL noise conditions below are INTRODUCED SIMULATIONS, not original data."]

base = make_pipe(); base.fit(Xtr, ytr)
b = scores(base, Xte, yte)
lines.append(f"Baseline clean RandomForest: ROC-AUC={b[0]:.4f} F1={b[1]:.4f} Accuracy={b[2]:.4f}")

# (a) label flips on TRAIN only
flip_res = {}
for pct in (0.10, 0.20):
    idx = rng.choice(len(ytr), size=int(pct * len(ytr)), replace=False)
    y_noisy = ytr.copy(); y_noisy.iloc[idx] = 1 - y_noisy.iloc[idx]
    m = make_pipe(); m.fit(Xtr, y_noisy); s = scores(m, Xte, yte)
    flip_res[pct] = s
    lines.append(f"SIMULATION label flip {int(pct*100)}% of train labels: ROC-AUC={s[0]:.4f} F1={s[1]:.4f} "
                 f"Accuracy={s[2]:.4f} (drop vs clean ROC-AUC {b[0]-s[0]:+.4f})")

# (b) Gaussian noise on scaled numeric features (train only)
pre_fit = base.named_steps["pre"].fit(Xtr)
Xtr_t = pre_fit.transform(Xtr); Xte_t = pre_fit.transform(Xte)
n_num = len(num)
Xtr_noisy = Xtr_t.copy()
Xtr_noisy[:, :n_num] = Xtr_noisy[:, :n_num] + rng.normal(0, 0.5, size=(len(Xtr_t), n_num))
m = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
m.fit(Xtr_noisy, ytr); p = m.predict_proba(Xte_t)[:, 1]
s = (roc_auc_score(yte, p), f1_score(yte, (p >= 0.5).astype(int), zero_division=0),
     accuracy_score(yte, (p >= 0.5).astype(int)))
lines.append(f"SIMULATION Gaussian noise N(0,0.5) added to scaled numeric train features: "
             f"ROC-AUC={s[0]:.4f} F1={s[1]:.4f} Accuracy={s[2]:.4f} (drop vs clean ROC-AUC {b[0]-s[0]:+.4f})")

# (c) corrupt 10% of categorical column Mjob (train only, random category swaps)
Xtr_c = Xtr.copy()
idx = rng.choice(len(Xtr_c), size=int(0.10 * len(Xtr_c)), replace=False)
cats = Xtr_c["Mjob"].unique()
Xtr_c.loc[idx, "Mjob"] = rng.choice(cats, size=len(idx))
m = make_pipe(); m.fit(Xtr_c, ytr); s = scores(m, Xte, yte)
lines.append(f"SIMULATION corrupt 10% of train 'Mjob' values (random category swaps): "
             f"ROC-AUC={s[0]:.4f} F1={s[1]:.4f} Accuracy={s[2]:.4f} (drop vs clean ROC-AUC {b[0]-s[0]:+.4f})")

# Fixes, evaluated against the 20% label-flip scenario
idx = rng.choice(len(ytr), size=int(0.20 * len(ytr)), replace=False)
y_noisy20 = ytr.copy(); y_noisy20.iloc[idx] = 1 - y_noisy20.iloc[idx]
noisy_base = flip_res[0.20]
# Fix 1: OOF-cleaned retrain - drop train rows whose OOF prediction disagrees with noisy label
oof = cross_val_predict(make_pipe(), Xtr, y_noisy20, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                        method="predict_proba")[:, 1]
keep = (oof >= 0.5).astype(int) == y_noisy20.values
m = make_pipe(); m.fit(Xtr[keep], y_noisy20[keep]); s = scores(m, Xte, yte)
lines.append(f"FIX 1 OOF-cleaned retrain on 20%-flipped labels (kept {keep.sum()}/{len(keep)} train rows): "
             f"ROC-AUC={s[0]:.4f} F1={s[1]:.4f} Accuracy={s[2]:.4f} "
             f"(vs noisy ROC-AUC {noisy_base[0]:.4f}: {'recovered' if s[0] > noisy_base[0] else 'did NOT recover'}; "
             f"vs clean {b[0]:.4f}: gap {b[0]-s[0]:+.4f})")
# Fix 2: regularised model (max_depth=8) on 20%-flipped labels
m = make_pipe({"max_depth": 8}); m.fit(Xtr, y_noisy20); s = scores(m, Xte, yte)
lines.append(f"FIX 2 regularised RandomForest (max_depth=8) on 20%-flipped labels: "
             f"ROC-AUC={s[0]:.4f} F1={s[1]:.4f} Accuracy={s[2]:.4f} "
             f"(vs noisy ROC-AUC {noisy_base[0]:.4f}: {'recovered' if s[0] > noisy_base[0] else 'did NOT recover'})")
# Fix 3: clip absences at train 95th percentile, clean split (robustness check)
cap = Xtr["absences"].quantile(0.95)
Xtr_clip = Xtr.copy(); Xte_clip = Xte.copy()
Xtr_clip["absences"] = Xtr_clip["absences"].clip(upper=cap); Xte_clip["absences"] = Xte_clip["absences"].clip(upper=cap)
m = make_pipe(); m.fit(Xtr_clip, ytr); s = scores(m, Xte_clip, yte)
lines.append(f"FIX 3 clip 'absences' at train 95th percentile ({cap:.0f}) on clean split: "
             f"ROC-AUC={s[0]:.4f} F1={s[1]:.4f} Accuracy={s[2]:.4f} (vs clean ROC-AUC {b[0]:.4f})")

(OUT / "results").mkdir(exist_ok=True); (OUT / "visualizations").mkdir(exist_ok=True)
(OUT / "results" / "results.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
labels = ["Clean", "Flip 10%", "Flip 20%", "Gauss noise", "Mjob corrupt"]
vals = [b[0], flip_res[0.10][0], flip_res[0.20][0], s[0] if False else 0, 0]
# recompute plot values cleanly from lines is fragile; use stored values:
vals = [b[0], flip_res[0.10][0], flip_res[0.20][0]]
plt.figure(figsize=(8, 4)); plt.bar(labels[:3], vals); plt.ylabel("Test ROC-AUC"); plt.title("Noise robustness: clean vs label-flip simulations")
plt.tight_layout(); plt.savefig(OUT / "visualizations" / "noise_robustness.png", dpi=150); plt.close()
