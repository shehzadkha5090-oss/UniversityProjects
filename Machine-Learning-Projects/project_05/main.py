"""Lab 5 — Custom Cross-Validation (UCI Student Performance, student-por.csv).

MANUAL cross-validation loops (cross_val_score is NOT used for the main
results): (a) shuffled 5-fold KFold, (b) stratified 5-fold KFold,
(c) LeaveOneGroupOut grouped by school (2 folds: train GP/test MS and
reverse). Model: LogisticRegression in the standard pipeline
(median/most-frequent imputation, scaling, one-hot). Metric: ROC-AUC per
fold, reported as mean +/- std. Target: pass = (G3 >= 10); G3 never a
feature; G1/G2 included by default. Full data: 649 rows (GP 423, MS 226).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, StratifiedKFold
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


def make_model():
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=1000, random_state=42))])


def run_scheme(name, splits, X, y):
    aucs = []
    for tr, te in splits:
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return aucs


def main():
    df = pd.read_csv(ROOT / "shared_dataset" / "dataset.csv", sep=";")
    y = (df["G3"] >= 10).astype(int)
    X = df[FEATURES]
    groups = df["school"]
    lines = [f"Dataset: UCI Student Performance (student-por.csv), {len(df)} rows",
             f"Target: pass = (G3 >= 10); pass {(y == 1).sum()} ({y.mean():.1%}), fail {(y == 0).sum()}",
             "Model: LogisticRegression (manual CV loops; cross_val_score not used).", ""]
    schemes = {
        "Shuffled 5-fold KFold": KFold(n_splits=5, shuffle=True, random_state=42).split(X),
        "Stratified 5-fold KFold": StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y),
        "LeaveOneGroupOut by school (2 folds)": LeaveOneGroupOut().split(X, y, groups),
    }
    results = {}
    for name, splits in schemes.items():
        aucs = run_scheme(name, splits, X, y)
        results[name] = aucs
        lines.append(f"{name}: folds={[f'{a:.4f}' for a in aucs]} "
                     f"mean={np.mean(aucs):.4f} std={np.std(aucs):.4f}")
    lines += ["",
              "Why the school-grouped score is much lower: each fold trains on one",
              "school and tests on a genuinely different school (GP pass 92.4% vs MS",
              "69.9%), so the distribution shift of Lab 3 hits it in full — the model",
              "has never seen the test school's student mix.",
              "Why the shuffled score is optimistic for deployment to a new school:",
              "shuffling mixes both schools into every train and test fold, so the",
              "test rows resemble the training rows; it estimates 'a new student from",
              "a known school', not 'a new school'. For deployment to an unseen",
              "school, the grouped (LeaveOneGroupOut) score is the honest estimate."]
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / "results.txt").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(results)
    means = [np.mean(v) for v in results.values()]
    stds = [np.std(v) for v in results.values()]
    ax.bar(names, means, yerr=stds, capsize=6, color=["#2980b9", "#27ae60", "#c0392b"])
    ax.set_ylabel("Mean ROC-AUC (+/- std)"); ax.set_ylim(0, 1.05)
    ax.set_title("Lab 5: Manual CV schemes — LogisticRegression")
    for bar, m, s in zip(ax.patches, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.02, f"{m:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "visualizations" / "cv_comparison.png", dpi=150)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
