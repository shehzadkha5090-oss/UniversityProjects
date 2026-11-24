"""Lab 4 — Explainability vs Performance (UCI Student Performance, student-por.csv).

Same stratified 75/25 split (seed 42; train 486, test 163) for three models:
LogisticRegression (interpretable), depth-4 DecisionTree (interpretable),
RandomForest with 200 trees (accurate but opaque). The forest is then
explained two ways: permutation importance on the test set (raw features,
permuted through the pipeline) and SHAP TreeExplainer on the first 100 test
rows (subsample stated here because full-test SHAP is slow). Target:
pass = (G3 >= 10); G3 never a feature; G1/G2 included by default.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
NUMERIC = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
           "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
           "G1", "G2"]
CATEGORICAL = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
               "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
               "nursery", "higher", "internet", "romantic"]
FEATURES = NUMERIC + CATEGORICAL


def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])


def main():
    df = pd.read_csv(ROOT / "shared_dataset" / "dataset.csv", sep=";")
    y = (df["G3"] >= 10).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], y, test_size=0.25, random_state=42, stratify=y)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree(max_depth=4)": DecisionTreeClassifier(max_depth=4, random_state=42),
        "RandomForest(200 trees)": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    lines = [f"Dataset: UCI Student Performance (student-por.csv), {len(df)} rows",
             f"Target: pass = (G3 >= 10); pass {(y == 1).sum()} ({y.mean():.1%}), fail {(y == 0).sum()}",
             f"Split: stratified 75/25, seed 42; train {len(X_train)}, test {len(X_test)}", ""]
    pipes = {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", make_pre()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        pipes[name] = pipe
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        acc = accuracy_score(y_test, pipe.predict(X_test))
        lines.append(f"{name}: ROC-AUC={auc:.4f} accuracy={acc:.4f}")
    # Permutation importance on the forest pipeline (raw feature names).
    rf_pipe = pipes["RandomForest(200 trees)"]
    perm = permutation_importance(rf_pipe, X_test, y_test, n_repeats=5,
                                  random_state=42, scoring="roc_auc")
    order = np.argsort(perm.importances_mean)[::-1]
    lines.append("")
    lines.append("Permutation importance (RandomForest, test set, ROC-AUC drop), top 10:")
    for i in order[:10]:
        lines.append(f"  {FEATURES[i]}: {perm.importances_mean[i]:.4f} "
                     f"(+/- {perm.importances_std[i]:.4f})")
    # SHAP on the forest itself, first 100 test rows (stated subsample).
    pre = make_pre().fit(X_train)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(pre.transform(X_train), y_train)
    X_sub = pre.transform(X_test.iloc[:100])
    feat_names = list(pre.get_feature_names_out())
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X_sub)
    sv1 = sv[1] if isinstance(sv, list) else sv[..., 1] if sv.ndim == 3 else sv
    lines.append("")
    lines.append("SHAP: TreeExplainer on RandomForest, first 100 test rows (subsample).")
    mean_abs = np.abs(sv1).mean(axis=0)
    top = np.argsort(mean_abs)[::-1][:10]
    lines.append("Top 10 features by mean |SHAP value|:")
    for i in top:
        lines.append(f"  {feat_names[i]}: {mean_abs[i]:.4f}")
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / "results.txt").write_text("\n".join(lines) + "\n")
    # Permutation plot (top 12).
    top12 = order[:12][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([FEATURES[i] for i in top12], perm.importances_mean[top12])
    ax.set_xlabel("Mean ROC-AUC drop when permuted (test set)")
    ax.set_title("Lab 4: Permutation importance — RandomForest (top 12)")
    fig.tight_layout()
    fig.savefig(OUT / "visualizations" / "permutation_importance.png", dpi=150)
    plt.close(fig)
    # SHAP summary (bar form, top 15, deterministic and readable).
    shap.summary_plot(sv1, X_sub, feature_names=feat_names, max_display=15,
                      plot_type="bar", show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.tight_layout()
    fig.savefig(OUT / "visualizations" / "shap_summary.png", dpi=150)
    plt.close(fig)
    lines.append("")
    lines.append("Trade-off: the forest's explanations (permutation + SHAP) agree that "
                 "earlier grades dominate, but only the simpler models can be read "
                 "directly as rules/coefficients; the forest buys accuracy at the "
                 "cost of needing post-hoc explanation. See README for the numbers.")
    (OUT / "results" / "results.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
