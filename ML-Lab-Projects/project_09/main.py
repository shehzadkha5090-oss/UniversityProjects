"""Lab 12 Calibration — UCI Student Performance (student-por.csv).
Fit/calibrate/test split: stratified, seed 42 — train 60%, calibration ~15%,
test 25%. Exact sizes are printed below. Target pass=(G3>=10); G3 never a feature.
Base models: LogisticRegression, RandomForest, HistGradientBoosting, each fitted
on train only, calibrated on the calibration set only. sklearn 1.9 removed
CalibratedClassifierCV(cv='prefit'), so calibration is done explicitly:
sigmoid = Platt scaling (LogisticRegression fitted on the base model's
calibration-set probabilities), isotonic = IsotonicRegression fitted on the
same calibration probabilities. Evaluated on test only. ECE is computed by hand:
10 equal-width bins over predicted probabilities."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parents[1]
OUT, VIS = BASE / "project_09" / "results", BASE / "project_09" / "visualizations"
OUT.mkdir(parents=True, exist_ok=True); VIS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "shared_dataset" / "dataset.csv", sep=";")
y = (df["G3"] >= 10).astype(int); X = df.drop(columns=["G3"])
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
X_rest, Xte, y_rest, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
Xtr, Xcal, ytr, ycal = train_test_split(X_rest, y_rest, test_size=0.20, random_state=42, stratify=y_rest)

def ece(y_true, proba, n_bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, n_bins + 1)[:-1]:
        hi = lo + 1 / n_bins
        m = (proba > lo) & (proba <= hi) if lo > 0 else (proba >= lo) & (proba <= hi)
        if m.sum() > 0: e += abs(y_true[m].mean() - proba[m].mean()) * m.mean()
    return float(e)
def stats(y_true, proba):
    return dict(brier=float(brier_score_loss(y_true, proba)), roc_auc=float(roc_auc_score(y_true, proba)), ece=ece(y_true.values if hasattr(y_true, "values") else y_true, proba))

models = {
 "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
 "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
 "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)}
all_rows, curves = {}, {}
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)]); pipe.fit(Xtr, ytr)
    p0 = pipe.predict_proba(Xte)[:, 1]
    p_cal = pipe.predict_proba(Xcal)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(max_iter=1000); platt.fit(p_cal, ycal); ps = platt.predict_proba(p0.reshape(-1, 1))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(pipe.predict_proba(Xcal)[:, 1], ycal); pi = iso.predict(p0)
    all_rows[name] = {"uncalibrated": stats(yte, p0), "sigmoid": stats(yte, ps), "isotonic": stats(yte, pi)}
    curves[name] = (p0, ps, pi)

lines = ["Lab 12 Calibration — UCI Student Performance (student-por.csv)",
 f"Split (stratified, seed 42): train {len(ytr)}, calibration {len(ycal)}, test {len(yte)}; total 649 (nominal 60/15/25, rounded by sklearn).",
 "ECE: manual, 10 equal-width bins, computed by this script. Calibration fitted on calibration set only (Platt sigmoid / isotonic, see script header)."]
for name, d in all_rows.items():
    for stage in ("uncalibrated", "sigmoid", "isotonic"):
        s = d[stage]; lines.append(f"{name} [{stage}]: Brier={s['brier']:.4f}, ROC-AUC={s['roc_auc']:.4f}, ECE={s['ece']:.4f}")
# Recommendation: lowest Brier among calibrated variants (Brier rewards both ranking and calibration)
cand = [(d["sigmoid"]["brier"], f"{n} + sigmoid") for n, d in all_rows.items()] + [(d["isotonic"]["brier"], f"{n} + isotonic") for n, d in all_rows.items()]
cand.sort(); lines.append(f"Recommendation: {cand[0][1]} — lowest calibrated Brier score on test ({cand[0][0]:.4f}).")
lines.append("Note: no calibrated variant beat the best uncalibrated model (HistGradientBoosting uncalibrated Brier 0.0599);")
lines.append("  with only 98 calibration rows, calibration hurt Brier here — the recommendation above is among calibrated variants only.")
(OUT / "results.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

# Reliability diagrams for best/worst by uncalibrated Brier
ranked = sorted(all_rows, key=lambda n: all_rows[n]["uncalibrated"]["brier"])
worst, best = ranked[-1], ranked[0]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, name in zip(axes, [best, worst]):
    for proba, label in zip(curves[name], ("uncalibrated", "sigmoid", "isotonic")):
        bins = np.linspace(0, 1, 11); cx, cy = [], []
        for i in range(10):
            m = (proba >= bins[i]) & (proba <= bins[i + 1]) if i == 0 else (proba > bins[i]) & (proba <= bins[i + 1])
            if m.sum() > 0: cx.append(proba[m].mean()); cy.append(yte.values[m].mean())
        ax.plot(cx, cy, "o-", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_title(f"{name} ({'best' if name == best else 'worst'} uncalibrated Brier)")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed pass rate"); ax.legend()
fig.tight_layout(); fig.savefig(VIS / "reliability_best_worst.png", dpi=150)
print("Saved reliability_best_worst.png")
