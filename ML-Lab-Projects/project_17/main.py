"""Lab 9: Time-Series Forecasting with Concept Drift (SIMULATION).

Dataset: UCI Student Performance (Portuguese, student-por.csv), 649 rows x 33 cols,
semicolon-delimited, loaded via Path(__file__).resolve().parents[1] / 'shared_dataset' / 'dataset.csv'.

CAVEAT: There is NO date/timestamp column and no true seasonal series in this
dataset, so this is a SIMULATION, not real time-series forecasting. File order
is used as a pseudo-time order only.

Design:
- 649 rows split into 7 sequential batches in file order: 6 x 100 + 49
  (batches 0-5 have 100 rows, batch 6 has 49 rows), the same batching as a
  streaming setup.
- Features: all columns except G3. Target: G3 (regression, primary) plus
  pass = (G3 >= 10) derived from it.
- Preprocessing: numeric median-impute + StandardScaler, categorical
  most-frequent-impute + OneHotEncoder(handle_unknown='ignore'), fit on data
  seen so far only. For the rolling model, fit on all batches before the
  forecast batch; for batch 0 there is no prior data, so forecasting starts
  at batch 1.
- (a) For each batch 1..6, forecast that batch's mean G3 and pass rate with:
    (i) a rolling-window baseline = previous batch's actual mean G3 / pass rate;
    (ii) a RandomForest regressor trained on all previous batches, predicting
        G3 per student, averaged to a batch mean and converted to a pass rate
        (fraction with predicted G3 >= 10).
- (b) Concept drift between consecutive batches: two-sample KS test on G1 and
  on G2 (threshold p < 0.05), plus the rolling pass-rate / mean-G3 shift.
  Comparison model: a RandomForest regressor trained once on batch 0 only,
  evaluated per batch 1..6, vs the rolling (retrained on all past data) model.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "shared_dataset" / "dataset.csv"
OUT_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR = Path(__file__).resolve().parent / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2",
]


def build_pipeline():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", numeric, NUMERIC_COLS),
            ("cat", categorical, None),  # placeholder, replaced below
        ], remainder="drop")),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])


def make_model(cat_cols):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", numeric, NUMERIC_COLS),
        ("cat", categorical, cat_cols),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])


def main():
    df = pd.read_csv(DATA_PATH, sep=";")
    assert df.shape == (649, 33), f"Unexpected dataset shape: {df.shape}"
    # G1/G2/G3 arrive as quoted strings in places; coerce all numeric cols.
    for c in NUMERIC_COLS + ["G3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    assert df["G3"].notna().all(), "G3 has missing values after coercion"

    feature_cols = [c for c in df.columns if c != "G3"]
    cat_cols = [c for c in feature_cols if c not in NUMERIC_COLS]

    # 7 sequential batches in file order: 6 x 100 + 49
    sizes = [100, 100, 100, 100, 100, 100, 49]
    batches, start = [], 0
    for s in sizes:
        batches.append(df.iloc[start:start + s].reset_index(drop=True))
        start += s
    assert sum(len(b) for b in batches) == 649

    rows = []  # per forecast batch 1..6
    batch0_model = make_model(cat_cols)
    batch0_model.fit(batches[0][feature_cols], batches[0]["G3"])

    for i in range(1, 7):
        prev, cur = batches[i - 1], batches[i]
        actual_mean = float(cur["G3"].mean())
        actual_pass = float((cur["G3"] >= 10).mean())
        base_mean = float(prev["G3"].mean())
        base_pass = float((prev["G3"] >= 10).mean())

        train = pd.concat(batches[:i], ignore_index=True)
        model = make_model(cat_cols)
        model.fit(train[feature_cols], train["G3"])
        pred = model.predict(cur[feature_cols])
        model_mean = float(np.mean(pred))
        model_pass = float((pred >= 10).mean())
        model_mae = float(mean_absolute_error(cur["G3"], pred))

        pred0 = batch0_model.predict(cur[feature_cols])
        mae0 = float(mean_absolute_error(cur["G3"], pred0))

        rows.append({
            "batch": i, "n": len(cur),
            "actual_mean": actual_mean, "base_mean": base_mean, "model_mean": model_mean,
            "model_mae": model_mae, "batch0_mae": mae0,
            "actual_pass": actual_pass, "base_pass": base_pass, "model_pass": model_pass,
        })

    r = pd.DataFrame(rows)
    base_mean_mae = float(np.mean(np.abs(r["base_mean"] - r["actual_mean"])))
    model_mean_mae = float(np.mean(np.abs(r["model_mean"] - r["actual_mean"])))
    base_pass_err = float(np.mean(np.abs(r["base_pass"] - r["actual_pass"])))
    model_pass_err = float(np.mean(np.abs(r["model_pass"] - r["actual_pass"])))
    rolling_mae_mean = float(r["model_mae"].mean())
    batch0_mae_mean = float(r["batch0_mae"].mean())
    rolling_better = int((r["model_mae"] < r["batch0_mae"]).sum())

    # (b) Drift: KS tests on G1 and G2 between consecutive batches + shifts
    drift = []
    for i in range(1, 7):
        prev, cur = batches[i - 1], batches[i]
        ks_g1 = ks_2samp(prev["G1"].dropna(), cur["G1"].dropna())
        ks_g2 = ks_2samp(prev["G2"].dropna(), cur["G2"].dropna())
        drift.append({
            "boundary": f"{i-1}->{i}",
            "ks_g1": float(ks_g1.statistic), "p_g1": float(ks_g1.pvalue),
            "ks_g2": float(ks_g2.statistic), "p_g2": float(ks_g2.pvalue),
            "mean_g3_shift": float(cur["G3"].mean() - prev["G3"].mean()),
            "pass_shift": float((cur["G3"] >= 10).mean() - (prev["G3"] >= 10).mean()),
            "drift_g1": bool(ks_g1.pvalue < 0.05),
            "drift_g2": bool(ks_g2.pvalue < 0.05),
        })
    d = pd.DataFrame(drift)
    drift_boundaries = [row["boundary"] for _, row in d.iterrows()
                        if row["drift_g1"] or row["drift_g2"]]

    # Plot: actual vs forecast mean G3 per batch (both methods)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = r["batch"].values
    ax.plot(x, r["actual_mean"], marker="o", label="Actual mean G3")
    ax.plot(x, r["base_mean"], marker="s", linestyle="--", label="Baseline forecast (prev. batch mean)")
    ax.plot(x, r["model_mean"], marker="^", linestyle="--", label="Rolling RandomForest forecast")
    ax.set_xlabel("Forecast batch (pseudo-time order)")
    ax.set_ylabel("Mean G3")
    ax.set_title("Lab 9 (Simulation): Actual vs Forecast Mean G3 per Batch")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = FIG_DIR / "actual_vs_forecast_mean_g3.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # Results text
    lines = []
    lines.append("Lab 9: Time-Series Forecasting with Concept Drift")
    lines.append("Dataset: UCI Student Performance (student-por.csv), 649 rows x 33 cols, semicolon-delimited")
    lines.append("")
    lines.append("CAVEAT: There is NO date/timestamp column and no true seasonal series in this")
    lines.append("dataset, so this is a SIMULATION, not real time-series forecasting. File order")
    lines.append("is used as a pseudo-time order only.")
    lines.append("")
    lines.append("Batching: 649 rows split into 7 sequential batches in file order (6 x 100 + 49),")
    lines.append("the same batching as a streaming setup. Forecasting starts at batch 1 because")
    lines.append("for batch 0 there is no prior data.")
    lines.append("Features: all columns except G3. Target: G3 (regression, primary) plus")
    lines.append("pass = (G3 >= 10) derived from it. Preprocessing: numeric median-impute +")
    lines.append("StandardScaler, categorical most-frequent-impute + OneHotEncoder(handle_unknown='ignore'),")
    lines.append("fit on data seen so far only.")
    lines.append("")
    lines.append("Per-batch forecasts (batches 1..6):")
    lines.append("batch  n  actual_mean_G3  baseline_forecast  model_forecast  model_per-student_MAE  batch0-only_MAE  actual_pass  baseline_pass  model_pass")
    for _, row in r.iterrows():
        lines.append(
            f"{int(row['batch'])}  {int(row['n'])}  {row['actual_mean']:.4f}  {row['base_mean']:.4f}  "
            f"{row['model_mean']:.4f}  {row['model_mae']:.4f}  {row['batch0_mae']:.4f}  "
            f"{row['actual_pass']:.4f}  {row['base_pass']:.4f}  {row['model_pass']:.4f}"
        )
    lines.append("")
    lines.append(f"Overall batch-mean forecast MAE (mean |forecast - actual mean G3|, batches 1..6):")
    lines.append(f"  Baseline (previous batch mean): {base_mean_mae:.4f}")
    lines.append(f"  Rolling RandomForest model:     {model_mean_mae:.4f}")
    lines.append(f"Overall pass-rate forecast error (mean |forecast - actual pass rate|, batches 1..6):")
    lines.append(f"  Baseline (previous batch pass rate): {base_pass_err:.4f}")
    lines.append(f"  Rolling RandomForest model:          {model_pass_err:.4f}")
    lines.append(f"Mean per-student G3 MAE, rolling model (batches 1..6): {rolling_mae_mean:.4f}")
    lines.append(f"Mean per-student G3 MAE, batch-0-only model (batches 1..6): {batch0_mae_mean:.4f}")
    lines.append(f"Rolling model beat batch-0-only model on {rolling_better} of 6 forecast batches (per-student MAE).")
    if rolling_mae_mean < batch0_mae_mean:
        lines.append("Verdict: YES, the rolling model (retrained on all past data) adapted better than")
        lines.append("the model trained once on batch 0 only, on average per-student MAE.")
    else:
        lines.append("Verdict: NO, the rolling model did NOT adapt better than the model trained once")
        lines.append("on batch 0 only, on average per-student MAE.")
    lines.append("")
    lines.append("Concept drift between consecutive batches:")
    lines.append("Method: two-sample Kolmogorov-Smirnov (KS) test on G1 and on G2 between consecutive")
    lines.append("batches, threshold p < 0.05 (drift flagged if p < 0.05 on either variable), plus the")
    lines.append("rolling pass-rate / mean-G3 shift (current batch minus previous batch).")
    lines.append("boundary  KS_G1  p_G1  drift_G1  KS_G2  p_G2  drift_G2  mean_G3_shift  pass_rate_shift")
    for _, row in d.iterrows():
        lines.append(
            f"{row['boundary']}  {row['ks_g1']:.4f}  {row['p_g1']:.4f}  {row['drift_g1']}  "
            f"{row['ks_g2']:.4f}  {row['p_g2']:.4f}  {row['drift_g2']}  "
            f"{row['mean_g3_shift']:+.4f}  {row['pass_shift']:+.4f}"
        )
    lines.append(f"Batch boundaries showing drift (p < 0.05 on G1 and/or G2): "
                 f"{', '.join(drift_boundaries) if drift_boundaries else 'none'}")
    lines.append("")
    lines.append(f"Plot: {png_path}")
    text = "\n".join(lines) + "\n"
    (OUT_DIR / "results.txt").write_text(text)
    print(text)
    print(f"PNG: {png_path} ({png_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
