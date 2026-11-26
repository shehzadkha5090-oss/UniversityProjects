"""Project 18 — Lab 11 Text Log Feature Engineering (ML Lab Projects v4).

Dataset: UCI Student Performance (student-por.csv), shared copy at
../shared_dataset/dataset.csv, 649 rows x 33 cols, semicolon-delimited.

Target: pass = (G3 >= 10). G3 is NEVER used as a feature, and neither
G3 nor the word 'pass' (derived from G3) appears in the constructed text.

CAVEAT: There are NO raw server logs in this dataset. The text in this
project is CONSTRUCTED from structured fields, not real logs, so this
demonstrates log-style text feature engineering technique, not real
log mining.

Models (same stratified 75/25 train/test split, seed 42, all Pipelines
fit on train only):
  (a) structured-features LogisticRegression baseline
  (b) TF-IDF (word n-grams 1-2, min_df=2) + LogisticRegression, text only
  (c) combined structured + TF-IDF model (ColumnTransformer concatenation)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATA_PATH = Path(__file__).resolve().parents[1] / "shared_dataset" / "dataset.csv"
RESULTS_DIR = BASE / "results"
VIZ_DIR = BASE / "visualizations"

CAVEAT = (
    "CAVEAT: There are NO raw server logs in this dataset. The text in this "
    "project is CONSTRUCTED from structured fields, not real logs, so this "
    "demonstrates log-style text feature engineering technique, not real "
    "log mining."
)

NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2",
]
CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]
# Every structured column except G3, in original dataset order.
TEXT_FIELDS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Keep the constructed text in the original column order for readability.
ORIGINAL_ORDER = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu",
    "Fedu", "Mjob", "Fjob", "reason", "guardian", "traveltime",
    "studytime", "failures", "schoolsup", "famsup", "paid",
    "activities", "nursery", "higher", "internet", "romantic",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2",
]


def build_text(row):
    """One short log-style description per student, from real values only.

    No invented facts, no G3, and no word derived from the target.
    """
    parts = []
    for col in ORIGINAL_ORDER:
        parts.append(f"{col} {row[col]}")
    return " ".join(parts)


def structured_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, proba), accuracy_score(y_test, pred)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, sep=";")
    assert df.shape == (649, 33), f"Unexpected dataset shape: {df.shape}"
    assert "G3" in df.columns
    # Target from G3; G3 itself is then dropped everywhere.
    y = (df["G3"] >= 10).astype(int)
    X_struct = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    # Ensure numeric columns are numeric (dataset stores some as strings).
    for col in NUMERIC_FEATURES:
        X_struct[col] = pd.to_numeric(X_struct[col], errors="coerce")
    text = df.apply(build_text, axis=1)
    assert "G3" not in " ".join(text.head(5).tolist())
    assert not any("pass" in t.lower().split() for t in text), \
        "constructed text must not contain the target word 'pass'"

    X_train_s, X_test_s, y_train, y_test, text_train, text_test = train_test_split(
        X_struct, y, text, test_size=0.25, random_state=42, stratify=y
    )
    assert len(X_train_s) == 486 and len(X_test_s) == 163

    # (a) Structured baseline, re-run here on this split.
    model_a = Pipeline([
        ("pre", structured_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model_a.fit(X_train_s, y_train)
    auc_a, acc_a = evaluate(model_a, X_test_s, y_test)

    # (b) Text only: TF-IDF word n-grams 1-2, min_df=2.
    model_b = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, token_pattern=r"(?u)\b\w+\b")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model_b.fit(text_train, y_train)
    auc_b, acc_b = evaluate(model_b, text_test, y_test)

    # (c) Combined: structured + TF-IDF concatenated via ColumnTransformer.
    combined_df_train = X_train_s.copy()
    combined_df_train["log_text"] = text_train.values
    combined_df_test = X_test_s.copy()
    combined_df_test["log_text"] = text_test.values
    combined_pre = ColumnTransformer([
        ("struct", structured_preprocessor(),
         NUMERIC_FEATURES + CATEGORICAL_FEATURES),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, token_pattern=r"(?u)\b\w+\b"), "log_text"),
    ])
    model_c = Pipeline([
        ("pre", combined_pre),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model_c.fit(combined_df_train, y_train)
    auc_c, acc_c = evaluate(model_c, combined_df_test, y_test)

    # Top TF-IDF terms by LogisticRegression coefficient (text-only model).
    tfidf = model_b.named_steps["tfidf"]
    coef = model_b.named_steps["clf"].coef_[0]
    terms = tfidf.get_feature_names_out()
    order = coef.argsort()
    top_negative = [(terms[i], float(coef[i])) for i in order[:10]]
    top_positive = [(terms[i], float(coef[i])) for i in order[::-1][:10]]

    # Example constructed text (first student), for the report.
    example_text = text.iloc[0]

    lines = []
    lines.append("Project 18 — Lab 11 Text Log Feature Engineering")
    lines.append("Dataset: UCI Student Performance (student-por.csv, shared_dataset/dataset.csv)")
    lines.append("Dataset shape: 649 rows x 33 columns, semicolon-delimited (sep=';')")
    lines.append("Target: pass = (G3 >= 10); pass 549 (84.6%), fail 100 (15.4%)")
    lines.append("Split: stratified 75/25 train/test, seed 42 (train 486, test 163)")
    lines.append(f"Test class counts: pass {(y_test == 1).sum()}, fail {(y_test == 0).sum()}")
    lines.append("")
    lines.append(CAVEAT)
    lines.append("")
    lines.append("Method: G3 was NEVER used as a feature, and neither G3 nor the word")
    lines.append("'pass' (derived from G3) appears in the constructed text. Each text")
    lines.append("description is built only from that student's actual structured field")
    lines.append("values (no invented facts). All Pipelines were fit on train only.")
    lines.append("")
    lines.append("Example constructed text (first student):")
    lines.append(example_text)
    lines.append("")
    lines.append("Test results (same split for all three models):")
    lines.append(f"(a) Structured LogisticRegression baseline: ROC-AUC = {auc_a:.4f}, accuracy = {acc_a:.4f}")
    lines.append(f"(b) TF-IDF (word n-grams 1-2, min_df=2) + LogisticRegression (text only): ROC-AUC = {auc_b:.4f}, accuracy = {acc_b:.4f}")
    lines.append(f"(c) Combined structured + TF-IDF LogisticRegression: ROC-AUC = {auc_c:.4f}, accuracy = {acc_c:.4f}")
    lines.append("")
    lines.append("Top 10 TF-IDF terms by LogisticRegression coefficient, text-only model (b):")
    lines.append("Top 10 POSITIVE coefficients (associated with pass):")
    for term, value in top_positive:
        lines.append(f"  {term}: {value:.4f}")
    lines.append("Top 10 NEGATIVE coefficients (associated with fail):")
    for term, value in top_negative:
        lines.append(f"  {term}: {value:.4f}")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Structured baseline: numeric median-impute + StandardScaler,")
    lines.append("  categorical most-frequent-impute + OneHotEncoder, LogisticRegression.")
    lines.append("- Combined model concatenates the structured preprocessor and the")
    lines.append("  TF-IDF vectorizer via ColumnTransformer, then LogisticRegression.")
    lines.append("- The structured baseline was re-run here for a fair same-split")
    lines.append("  comparison; no earlier number was copied.")
    results_text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "results.txt").write_text(results_text, encoding="utf-8")

    # ROC-AUC comparison bar chart.
    labels = ["(a) Structured", "(b) TF-IDF text only", "(c) Combined"]
    values = [auc_a, auc_b, auc_c]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Project 18: Test ROC-AUC comparison")
    ax.set_ylim(0, 1.0)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    png_path = VIZ_DIR / "roc_auc_comparison.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(results_text)
    print(f"Saved: {RESULTS_DIR / 'results.txt'}")
    print(f"Saved: {png_path}")

    # README with the ACTUAL numbers (copied from the run above).
    readme = f"""# Project 18 — Lab 11 Text Log Feature Engineering

## Dataset
UCI Student Performance dataset (`student-por.csv`, shared copy at
`../shared_dataset/dataset.csv`): 649 rows x 33 columns, semicolon-delimited
(read with `sep=';'`). Only `student-por.csv` was used — not `student-mat.csv`
or any other dataset.

## Target
`pass = (G3 >= 10)`: pass 549 (84.6%), fail 100 (15.4%).

Split: stratified 75/25 train/test, seed 42 (train 486, test 163).

## Caveat
{CAVEAT}

## Method
G3 was NEVER used as a feature, and neither G3 nor the word 'pass' (derived
from G3) appears in the constructed text. One short log-style text
description was built per student from that student's actual structured
field values only (no invented facts), for example:

> {example_text}

Three models were compared on the same split, all Pipelines fit on train only:

- (a) Structured-features LogisticRegression baseline: numeric median-impute +
  StandardScaler, categorical most-frequent-impute + OneHotEncoder.
- (b) TF-IDF (word n-grams 1-2, `min_df=2`) + LogisticRegression on the text only.
- (c) Combined structured + TF-IDF model (ColumnTransformer concatenation,
  then LogisticRegression).

The structured baseline was re-run here for a fair same-split comparison;
no earlier number was copied.

## Results (test set)
- (a) Structured LogisticRegression baseline: ROC-AUC = {auc_a:.4f}, accuracy = {acc_a:.4f}
- (b) TF-IDF (word n-grams 1-2, min_df=2) + LogisticRegression (text only): ROC-AUC = {auc_b:.4f}, accuracy = {acc_b:.4f}
- (c) Combined structured + TF-IDF LogisticRegression: ROC-AUC = {auc_c:.4f}, accuracy = {acc_c:.4f}

## Top TF-IDF terms (text-only model, by LogisticRegression coefficient)
Top 10 POSITIVE coefficients (associated with pass):
{chr(10).join(f"- `{term}`: {value:.4f}" for term, value in top_positive)}

Top 10 NEGATIVE coefficients (associated with fail):
{chr(10).join(f"- `{term}`: {value:.4f}" for term, value in top_negative)}

## Files
- `main.py` — self-contained script (loads the dataset via
  `Path(__file__).resolve().parents[1] / 'shared_dataset' / 'dataset.csv'`)
- `results/results.txt` — full results
- `visualizations/roc_auc_comparison.png` — ROC-AUC comparison bar chart
"""
    (BASE / "README.md").write_text(readme, encoding="utf-8")
    print(f"Saved: {BASE / 'README.md'}")


if __name__ == "__main__":
    main()
