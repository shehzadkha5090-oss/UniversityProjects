"""Lab 3 — Distribution Shift (UCI Student Performance, student-por.csv).

Train on one school, test on the other, and reverse. The two schools differ
sharply: GP pass rate 92.4% (391/423) vs MS pass rate 69.9% (158/226), so a
model tuned on one school degrades on the other.
Shift is measured per feature between GP and MS: KS statistic for numeric
features, and max category-proportion difference for categorical features.
PSI (10 quantile bins, from the training school) is also reported for numeric
features. Stability rule, stated in advance: a feature is 'stable' if
KS < 0.20 (numeric) or max proportion difference < 0.20 (categorical).
A stable-only model is then retrained and judged honestly on the held-out
school. Target: pass = (G3 >= 10); G3 never a feature; G1/G2 included by
default. RandomForest, 200 trees.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
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


def make_model(cols):
    num = [c for c in cols if c in NUMERIC]
    cat = [c for c in cols if c in CATEGORICAL]
    steps = []
    if num:
        steps.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler())]), num))
    if cat:
        steps.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return Pipeline([("pre", ColumnTransformer(steps)),
                     ("clf", RandomForestClassifier(n_estimators=200, random_state=42))])


def psi(expected, actual, bins=10):
    qs = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    e_perc = np.histogram(expected, bins=qs)[0] / len(expected)
    a_perc = np.histogram(actual, bins=qs)[0] / len(actual)
    e_perc = np.where(e_perc == 0, 0.0001, e_perc)
    a_perc = np.where(a_perc == 0, 0.0001, a_perc)
    return float(np.sum((e_perc - a_perc) * np.log(e_perc / a_perc)))


def main():
    df = pd.read_csv(ROOT / "shared_dataset" / "dataset.csv", sep=";")
    y = (df["G3"] >= 10).astype(int)
    gp, ms = df["school"] == "GP", df["school"] == "MS"
    lines = [f"Dataset: UCI Student Performance (student-por.csv), {len(df)} rows",
             f"GP: {gp.sum()} students, pass rate {(y[gp]).mean():.1%} (391/423)",
             f"MS: {ms.sum()} students, pass rate {(y[ms]).mean():.1%} (158/226)",
             "Shift statistics (GP vs MS):"]
    stable = []
    for col in NUMERIC:
        ks = ks_2samp(df.loc[gp, col], df.loc[ms, col]).statistic
        p = psi(df.loc[gp, col].values, df.loc[ms, col].values)
        ok = ks < 0.20
        stable.append(col) if ok else None
        lines.append(f"  {col}: KS={ks:.3f} PSI={p:.3f} stable={ok}")
    for col in CATEGORICAL:
        if col == "school":
            continue
        pg = df.loc[gp, col].value_counts(normalize=True)
        pm = df.loc[ms, col].value_counts(normalize=True)
        diff = max(abs(pg.get(k, 0) - pm.get(k, 0))
                   for k in set(pg.index) | set(pm.index))
        ok = diff < 0.20
        stable.append(col) if ok else None
        lines.append(f"  {col}: max_prop_diff={diff:.3f} stable={ok}")
    lines.append(f"Stable features (KS/prop-diff < 0.20): {len(stable)} of 31: {stable}")
    all_feats = [c for c in df.columns if c not in ("G3", "school")]
    results = {}
    for direction, tr, te in (("train GP -> test MS", gp, ms), ("train MS -> test GP", ms, gp)):
        for label, cols in (("all features", all_feats), ("stable only", stable)):
            m = make_model(cols)
            m.fit(df.loc[tr, cols], y[tr])
            auc = roc_auc_score(y[te], m.predict_proba(df.loc[te, cols])[:, 1])
            results[(direction, label)] = auc
            lines.append(f"{direction} | {label}: ROC-AUC={auc:.4f} "
                         f"(train {tr.sum()}, test {te.sum()})")
    for direction in ("train GP -> test MS", "train MS -> test GP"):
        d = results[(direction, "stable only")] - results[(direction, "all features")]
        lines.append(f"{direction}: stable-only change = {d:+.4f} ROC-AUC "
                     f"({'helped' if d > 0 else 'did NOT help' if d < 0 else 'no change'})")
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / "results.txt").write_text("\n".join(lines) + "\n")
    labels = ["GP->MS\nall", "GP->MS\nstable", "MS->GP\nall", "MS->GP\nstable"]
    vals = [results[("train GP -> test MS", "all features")],
            results[("train GP -> test MS", "stable only")],
            results[("train MS -> test GP", "all features")],
            results[("train MS -> test GP", "stable only")]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, vals, color=["#2980b9", "#27ae60", "#2980b9", "#27ae60"])
    ax.set_ylabel("Held-out-school ROC-AUC"); ax.set_ylim(0, 1.05)
    ax.set_title("Lab 3: Cross-school performance, all vs stable-only features")
    for bar, v in zip(ax.patches, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "visualizations" / "cross_school_roc_auc.png", dpi=150)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
