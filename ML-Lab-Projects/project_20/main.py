"""Lab 20 — End-to-End System Design (UCI Student Performance, student-por).

Pipeline: ingest -> validate -> preprocess -> train -> evaluate -> serve
(artefact) -> monitor (drift check) -> retrain (trigger rule).

Target: pass = G3 >= 10. G3 itself is NEVER used as a feature.
Dataset: shared_dataset/dataset.csv (student-por.csv, semicolon-delimited).
Split: stratified 75/25, seed 42 -> train 486, test 163.
Model: HistGradientBoostingClassifier (see results/README for justification).
Run with: ~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATA_PATH = Path(__file__).resolve().parents[1] / "shared_dataset" / "dataset.csv"
RESULTS = BASE / "results"
VIZ = BASE / "visualizations"
RESULTS.mkdir(parents=True, exist_ok=True)
VIZ.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- ingest
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, sep=";")
print(f"Loaded shape: {df.shape}")

# ------------------------------------------------------------- validate
# Fail loudly if any check fails.
assert df.shape[0] == 649, f"Row count check FAILED: expected 649, got {df.shape[0]}"
assert df.shape[1] == 33, f"Column count check FAILED: expected 33, got {df.shape[1]}"
n_missing = int(df.isna().sum().sum())
assert n_missing == 0, f"Missing-value check FAILED: {n_missing} missing values"
assert df["G3"].min() >= 0 and df["G3"].max() <= 20, (
    f"G3 range check FAILED: min={df['G3'].min()}, max={df['G3'].max()}"
)
pass_rate = float((df["G3"] >= 10).mean())
assert 0.84 <= pass_rate <= 0.85, f"Pass-rate check FAILED: {pass_rate:.4f}"
print("VALIDATION PASSED:")
print(f"  rows = {df.shape[0]}, columns = {df.shape[1]}, missing = {n_missing}")
print(f"  G3 range = {df['G3'].min()}-{df['G3'].max()} (allowed 0-20)")
print(f"  pass rate (G3>=10) = {pass_rate:.4f} ({pass_rate*100:.2f}%, about 84.6%)")

# ------------------------------------------------------- target / split
# NEVER use G3 as a feature: it defines the target.
y = (df["G3"] >= 10).astype(int)  # 1 = pass, 0 = fail
X = df.drop(columns=["G3"])
assert "G3" not in X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
assert len(X_train) == 486 and len(X_test) == 163, (
    f"Split check FAILED: train={len(X_train)}, test={len(X_test)}"
)
print(f"Split (stratified, seed 42): train = {len(X_train)}, test = {len(X_test)}")
print(f"  train pass rate = {y_train.mean():.4f}, test pass rate = {y_test.mean():.4f}")

# ------------------------------------------------------------ preprocess
numeric_features = list(X.select_dtypes(include="number").columns)
categorical_features = list(X.select_dtypes(exclude="number").columns)
print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features),
])

# ---------------------------------------------------------------- train
# Chosen model: HistGradientBoostingClassifier — best single model in
# project_12 (ROC-AUC 0.9597 vs LogisticRegression 0.9420, RandomForest
# 0.9432, KNN 0.7996 on the same split protocol). Re-fit here from scratch.
model = Pipeline([
    ("preprocess", preprocess),
    ("clf", HistGradientBoostingClassifier(random_state=42)),
])
model.fit(X_train, y_train)  # preprocessing is fit on train only
print("Model trained: HistGradientBoostingClassifier (pipeline fit on train only)")

# -------------------------------------------------------------- evaluate
proba = model.predict_proba(X_test)[:, 1]
pred = model.predict(X_test)
roc_auc = float(roc_auc_score(y_test, proba))
accuracy = float(accuracy_score(y_test, pred))
f1_fail = float(f1_score(y_test, pred, pos_label=0))
print("HELD-OUT TEST (n=163):")
print(f"  ROC-AUC = {roc_auc:.4f}")
print(f"  Accuracy = {accuracy:.4f}")
print(f"  Fail-class F1 (label 0) = {f1_fail:.4f}")

# ------------------------------------------------------- save + verify
model_path = RESULTS / "model.joblib"
joblib.dump(model, model_path)
assert model_path.exists() and model_path.stat().st_size > 0, "model.joblib is empty/missing"
print(f"Saved model to {model_path} ({model_path.stat().st_size} bytes)")

reloaded = joblib.load(model_path)
reload_pred = reloaded.predict(X_test.iloc[:5])
reload_proba = reloaded.predict_proba(X_test.iloc[:5])[:, 1]
assert len(reload_pred) == 5 and reload_proba.shape == (5,)
print(f"Reload verification OK: reloaded model predicted on 5 rows -> {list(reload_pred)}")

# ------------------------------------------- example predictions (serve)
print("EXAMPLE PREDICTIONS — first 3 test rows:")
examples = []
for i in range(3):
    row = X_test.iloc[[i]]
    p = float(reloaded.predict_proba(row)[0, 1])
    lab = int(reloaded.predict(row)[0])
    true = int(y_test.iloc[i])
    examples.append((i, p, lab, true))
    print(f"  test row {i}: P(pass)={p:.4f}, predicted={'pass' if lab == 1 else 'fail'}, "
          f"true={'pass' if true == 1 else 'fail'}")

# ------------------------------------------------------------ drift check
# Rule (stated before seeing the verdict): retraining trigger FIRES if
# any KS statistic > 0.20, or the mean KS across checked features > 0.10.
drift_features = ["G1", "G2", "absences"]
ks_results = {}
for feat in drift_features:
    stat = float(ks_2samp(X_train[feat], X_test[feat]).statistic)
    ks_results[feat] = stat
    print(f"  KS {feat}: train vs test = {stat:.4f}")
mean_ks = float(sum(ks_results.values()) / len(ks_results))
trigger = any(v > 0.20 for v in ks_results.values()) or mean_ks > 0.10
verdict = "RETRAINING TRIGGER FIRES" if trigger else "NO RETRAINING TRIGGER"
print(f"Drift rule: fire if any KS > 0.20 or mean KS > 0.10; mean KS = {mean_ks:.4f}")
print(f"Drift verdict: {verdict}")

# ------------------------------------------------- architecture diagram
fig, ax = plt.subplots(figsize=(12, 3.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")
stages = [
    "1. Ingest\nshared CSV\n(sep=';')",
    "2. Validate\n649 rows, 0 NA\nG3 0-20",
    "3. Preprocess\nmedian/scale +\none-hot (train only)",
    "4. Train\nHistGradient\nBoosting",
    "5. Evaluate\nROC-AUC / acc /\nfail F1 (test n=163)",
    "6. Serve\nmodel.joblib\npredict_proba",
    "7. Monitor\nKS drift, pass\nrate, metrics",
]
for i, label in enumerate(stages):
    x = 0.15 + i * 1.40
    ax.add_patch(plt.Rectangle((x, 1.0), 1.25, 1.2, facecolor="#dbeafe",
                               edgecolor="#1e40af", linewidth=1.5))
    ax.text(x + 0.625, 1.6, label, ha="center", va="center", fontsize=8)
    if i < len(stages) - 1:
        ax.annotate("", xy=(x + 1.38, 1.6), xytext=(x + 1.27, 1.6),
                    arrowprops=dict(arrowstyle="->", color="#1e40af", lw=1.5))
# Retrain loop-back arrow: monitor -> train
ax.annotate("retrain if KS trigger fires", xy=(4.65, 1.0), xytext=(8.6, 0.45),
            ha="center", fontsize=8, color="#b91c1c",
            arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.5,
                            connectionstyle="arc3,rad=0.25"))
ax.set_title("Lab 20 — End-to-End System: ingest -> validate -> preprocess -> "
             "train -> evaluate -> serve -> monitor -> retrain", fontsize=11)
png_path = VIZ / "architecture.png"
fig.tight_layout()
fig.savefig(png_path, dpi=150)
plt.close(fig)
assert png_path.exists() and png_path.stat().st_size > 0, "architecture.png empty/missing"
print(f"Saved architecture diagram to {png_path} ({png_path.stat().st_size} bytes)")

# ------------------------------------------------------------ results.txt
lines = [
    "Lab 20 — End-to-End System Design (UCI Student Performance, student-por)",
    f"Dataset: shared_dataset/dataset.csv (semicolon-delimited), shape {df.shape}",
    "Validation: rows=649, missing=0, "
    f"G3 range={df['G3'].min()}-{df['G3'].max()} (allowed 0-20), "
    f"pass rate (G3>=10)={pass_rate:.4f} ({pass_rate*100:.2f}%)",
    f"Split: stratified 75/25, seed 42, train={len(X_train)}, test={len(X_test)}",
    "Model: HistGradientBoostingClassifier in a Pipeline "
    "(numeric median-impute + StandardScaler; categorical most-frequent-impute "
    "+ OneHotEncoder(handle_unknown='ignore')); fit on train only; G3 never a feature",
    f"Test ROC-AUC = {roc_auc:.4f}",
    f"Test accuracy = {accuracy:.4f}",
    f"Test fail-class F1 (label 0) = {f1_fail:.4f}",
    f"Saved model: results/model.joblib ({model_path.stat().st_size} bytes), "
    "reload-verified (predicted on 5 test rows)",
]
for i, p, lab, true in examples:
    lines.append(
        f"Example test row {i}: P(pass)={p:.4f}, "
        f"predicted={'pass' if lab == 1 else 'fail'}, "
        f"true={'pass' if true == 1 else 'fail'}"
    )
for feat, stat in ks_results.items():
    lines.append(f"Drift KS {feat} (train vs test) = {stat:.4f}")
lines.append(
    f"Drift rule: fire if any KS > 0.20 or mean KS > 0.10; mean KS = {mean_ks:.4f}; "
    f"verdict: {verdict}"
)
lines.append(f"Architecture diagram: visualizations/architecture.png "
             f"({png_path.stat().st_size} bytes)")
results_path = RESULTS / "results.txt"
results_path.write_text("\n".join(lines) + "\n")
print(f"Saved results to {results_path}")
print("DONE — all artefacts verified non-empty.")
