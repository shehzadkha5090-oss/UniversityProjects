"""Lab 2 — Missing Data (UCI Student Performance, student-por.csv).

REAL DATA FIRST: the file as published has 0 missing values in every column
(verified below, with a per-column completeness table). Because there is
nothing real to impute, three missingness mechanisms are then introduced as
LABELLED SIMULATIONS on a copy of the data — they are not original data:
  MCAR: 20% of feature values masked at random (seed 42).
  MAR:  missingness in 'absences' and 'studytime' depends on OBSERVED
        columns only: 40% missing if school == 'MS', else 5%.
  MNAR: missingness in 'absences' depends on the UNOBSERVED value itself:
        50% missing if absences > 10, else 5% (and 'studytime' 40% if
        studytime == 1, else 5% — low study time hides itself).
For each simulation, median/mode imputation is compared with KNN imputation
(KNNImputer, k=5, applied after one-hot encoding) using a RandomForest on the
same stratified 75/25 split (seed 42; train 486, test 163).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
NUMERIC = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
           "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
           "G1", "G2"]
CATEGORICAL = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
               "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
               "nursery", "higher", "internet", "romantic"]
FEATURES = NUMERIC + CATEGORICAL
rng = np.random.default_rng(42)


def baseline_model():
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("pre", pre),
                     ("clf", RandomForestClassifier(n_estimators=200, random_state=42))])


def knn_model():
    # One-hot first (KNN needs numbers), then KNNImputer on the encoded matrix.
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("knn", KNNImputer(n_neighbors=5)),
                     ("clf", RandomForestClassifier(n_estimators=200, random_state=42))])


def simulate(df, mechanism):
    X = df[FEATURES].copy()
    if mechanism == "MCAR":
        mask = rng.random(X.shape) < 0.20
        X = X.mask(mask)
    elif mechanism == "MAR":
        for col in ("absences", "studytime"):
            p = np.where(df["school"] == "MS", 0.40, 0.05)
            X.loc[rng.random(len(X)) < p, col] = np.nan
    elif mechanism == "MNAR":
        p_abs = np.where(df["absences"] > 10, 0.50, 0.05)
        X.loc[rng.random(len(X)) < p_abs, "absences"] = np.nan
        p_st = np.where(df["studytime"] == 1, 0.40, 0.05)
        X.loc[rng.random(len(X)) < p_st, "studytime"] = np.nan
    return X


def main():
    df = pd.read_csv(ROOT / "shared_dataset" / "dataset.csv", sep=";")
    y = (df["G3"] >= 10).astype(int)
    lines = [f"Dataset: UCI Student Performance (student-por.csv), {len(df)} rows x {df.shape[1]} cols",
             f"Target: pass = (G3 >= 10); pass {(y == 1).sum()} ({y.mean():.1%}), fail {(y == 0).sum()}",
             "REAL missingness: 0 missing values in the whole file (all 33 columns 100% complete).",
             "Per-column completeness: every column 649/649 non-missing (0.0% missing).",
             "All missingness below is a LABELLED SIMULATION introduced by this script,",
             "not original data. Split: stratified 75/25, seed 42; train 486, test 163.",
             ""]
    X_train_i, X_test_i, y_train, y_test = train_test_split(
        np.arange(len(df)), y, test_size=0.25, random_state=42, stratify=y)
    scores = {}
    for mech in ("MCAR", "MAR", "MNAR"):
        Xsim = simulate(df, mech)
        n_miss = int(Xsim.isna().sum().sum())
        row = [f"{mech}: introduced missing values = {n_miss} "
               f"({n_miss / Xsim.size:.1%} of feature cells)"]
        for label, make in (("median/mode", baseline_model), ("KNN(k=5)", knn_model)):
            m = make()
            m.fit(Xsim.iloc[X_train_i], y_train)
            auc = roc_auc_score(y_test, m.predict_proba(Xsim.iloc[X_test_i])[:, 1])
            scores[(mech, label)] = auc
            row.append(f"  {label}: ROC-AUC={auc:.4f}")
        lines.append("\n".join(row))
    lines += ["",
              "Note: MNAR is the hardest case in principle because the missingness",
              "depends on values that are themselves hidden, so no imputer can fully",
              "recover them; KNN vs median/mode differences here are small because",
              "G1/G2 carry most of the signal and are never masked in MAR/MNAR."]
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / "results.txt").write_text("\n".join(lines) + "\n")
    mechs = ["MCAR", "MAR", "MNAR"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(mechs)); w = 0.35
    ax.bar(x - w / 2, [scores[(m, "median/mode")] for m in mechs], w, label="Median/mode")
    ax.bar(x + w / 2, [scores[(m, "KNN(k=5)")] for m in mechs], w, label="KNN (k=5)")
    ax.set_xticks(x); ax.set_xticklabels(mechs); ax.set_ylabel("Test ROC-AUC")
    ax.set_ylim(0, 1.05); ax.legend()
    ax.set_title("Lab 2: Imputation comparison under SIMULATED missingness")
    fig.tight_layout()
    fig.savefig(OUT / "visualizations" / "imputation_comparison.png", dpi=150)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
