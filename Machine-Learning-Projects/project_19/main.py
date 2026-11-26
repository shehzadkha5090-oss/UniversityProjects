"""Lab 13 — Causal Inference (observational) on the UCI Student Performance data.

Treatment: schoolsup (extra educational support, yes/no).
Outcome:  pass = (G3 >= 10).  G3 is NEVER used as a feature (it defines the outcome).
Dataset: ../shared_dataset/dataset.csv  (student-por.csv, 649 rows x 33 cols, ';'-delimited).

Methods:
  (1) Propensity score: LogisticRegression on stated confounders, with
      one-hot encoding for categoricals and StandardScaler for numerics.
  (2) IPW estimate of the ATE on pass, with a bootstrap standard error
      (200 resamples, seed 42).
  (3) Greedy nearest-neighbour matching on the propensity logit, caliper
      0.2 SD of the logit, without replacement.
  (4) Covariate balance: standardised mean differences (SMD) before and
      after weighting / matching.

CAVEAT: there is NO randomised treatment here. This is an OBSERVATIONAL
analysis and cannot prove causality. The result must NOT be read as
"support causes passing".
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parents[1] / "shared_dataset" / "dataset.csv"
RESULTS_DIR = BASE / "results"
VIZ_DIR = BASE / "visualizations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

CONFOUNDERS_CAT = ["school", "sex"]
CONFOUNDERS_NUM = ["age", "Medu", "Fedu", "studytime", "failures", "G1"]
CONFOUNDERS = CONFOUNDERS_CAT + CONFOUNDERS_NUM
BOOTSTRAP_N = 200
BOOTSTRAP_SEED = 42
CALIPER_SD = 0.2


def make_pipeline():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CONFOUNDERS_CAT),
        ("num", StandardScaler(), CONFOUNDERS_NUM),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
    ])


def fit_ps(X, t):
    pipe = make_pipeline()
    pipe.fit(X, t)
    ps = pipe.predict_proba(X)[:, 1]
    return np.clip(ps, 0.01, 0.99), pipe


def ipw_ate(t, y, ps):
    return float(np.mean(t * y / ps - (1 - t) * y / (1 - ps)))


def greedy_match(t, y, ps):
    logit = np.log(ps / (1 - ps))
    sd = float(np.std(logit, ddof=1))
    caliper = CALIPER_SD * sd
    treated_idx = np.where(t == 1)[0]
    control_idx = np.where(t == 0)[0]
    used = set()
    pairs = []
    for ti in treated_idx:
        # nearest available control on the logit
        dists = np.abs(logit[control_idx] - logit[ti])
        order = np.argsort(dists)
        for oi in order:
            ci = control_idx[oi]
            if dists[oi] <= caliper and ci not in used:
                used.add(ci)
                pairs.append((ti, ci))
                break
    if pairs:
        matched_ate = float(np.mean([y[ti] - y[ci] for ti, ci in pairs]))
    else:
        matched_ate = float("nan")
    return pairs, matched_ate, sd, caliper


def smd(a, b):
    """Standardised mean difference between two groups (pooled SD denominator)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def weighted_mean(values, weights):
    return float(np.sum(values * weights) / np.sum(weights))


def main():
    df = pd.read_csv(DATA, sep=";")
    assert df.shape == (649, 33), f"Unexpected dataset shape {df.shape}"
    df["G3"] = pd.to_numeric(df["G3"])
    df["G1"] = pd.to_numeric(df["G1"])
    t = (df["schoolsup"].astype(str).str.strip().str.lower() == "yes").astype(int).to_numpy()
    y = (df["G3"] >= 10).astype(int).to_numpy()
    X = df[CONFOUNDERS].copy()

    n_t = int(t.sum())
    n_c = int((1 - t).sum())
    rate_t = float(y[t == 1].mean())
    rate_c = float(y[t == 0].mean())
    raw_diff_pp = (rate_t - rate_c) * 100

    # (1) Propensity scores
    ps, _pipe = fit_ps(X, t)

    # (2) IPW + bootstrap SE (refit the whole pipeline inside each resample)
    ate_ipw = ipw_ate(t, y, ps)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = []
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, len(df), len(df))
        try:
            ps_b, _ = fit_ps(X.iloc[idx], t[idx])
            boot.append(ipw_ate(t[idx], y[idx], ps_b))
        except Exception:
            continue
    boot = np.array(boot)
    se_ipw = float(np.std(boot, ddof=1)) if len(boot) > 1 else float("nan")
    ci_lo, ci_hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) \
        if len(boot) else (float("nan"), float("nan"))

    # (3) Greedy nearest-neighbour matching on the logit
    pairs, matched_ate, logit_sd, caliper = greedy_match(t, y, ps)
    n_matched = len(pairs)
    matched_rate_t = float(np.mean([y[ti] for ti, _ in pairs])) if pairs else float("nan")
    matched_rate_c = float(np.mean([y[ci] for _, ci in pairs])) if pairs else float("nan")

    # (4) Balance: SMD before, after IPW weighting, after matching.
    # Balance variables: failures, G1, Medu (numeric) + school (as GP indicator).
    balance_vars = {
        "failures": df["failures"].to_numpy(dtype=float),
        "G1": df["G1"].to_numpy(dtype=float),
        "Medu": df["Medu"].to_numpy(dtype=float),
        "school (GP=1)": (df["school"].astype(str) == "GP").astype(float).to_numpy(),
    }
    w_t = t / ps
    w_c = (1 - t) / (1 - ps)
    rows = []
    for name, v in balance_vars.items():
        before = smd(v[t == 1], v[t == 0])
        # Weighted SMD: weighted means, unweighted pooled SD (standard convention)
        pooled = np.sqrt((np.var(v[t == 1], ddof=1) + np.var(v[t == 0], ddof=1)) / 2.0)
        wmean_t = weighted_mean(v[t == 1], w_t[t == 1])
        wmean_c = weighted_mean(v[t == 0], w_c[t == 0])
        after_w = float((wmean_t - wmean_c) / pooled) if pooled else 0.0
        if pairs:
            after_m = smd(np.array([v[ti] for ti, _ in pairs]),
                          np.array([v[ci] for _, ci in pairs]))
        else:
            after_m = float("nan")
        rows.append((name, before, after_w, after_m))
    balance = pd.DataFrame(rows, columns=["variable", "smd_before", "smd_ipw", "smd_matched"])

    # ---- Plots ----
    # SMD before/after
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = balance["variable"].tolist()
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, balance["smd_before"].abs(), w, label="Before adjustment")
    ax.bar(x, balance["smd_ipw"].abs(), w, label="After IPW weighting")
    ax.bar(x + w, balance["smd_matched"].abs(), w, label="After matching")
    ax.axhline(0.1, color="red", linestyle="--", linewidth=1, label="|SMD| = 0.1 guide")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("|Standardised mean difference|")
    ax.set_title("Covariate balance: before vs after adjustment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "smd_before_after.png", dpi=150)
    plt.close(fig)

    # Propensity overlap histogram
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(ps[t == 0], bins=20, alpha=0.6, label=f"No schoolsup (n={n_c})")
    ax.hist(ps[t == 1], bins=20, alpha=0.6, label=f"schoolsup = yes (n={n_t})")
    ax.set_xlabel("Estimated propensity score P(schoolsup = yes | confounders)")
    ax.set_ylabel("Count")
    ax.set_title("Propensity-score overlap (common support check)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "propensity_overlap.png", dpi=150)
    plt.close(fig)

    # ---- Results ----
    lines = []
    lines.append("Lab 13 — Causal Inference (OBSERVATIONAL)")
    lines.append("Dataset: UCI Student Performance, student-por.csv (shared_dataset/dataset.csv),")
    lines.append("  649 rows x 33 columns, semicolon-delimited. Outcome and treatment defined below.")
    lines.append("")
    lines.append("CAVEAT — READ FIRST:")
    lines.append("  There is NO randomised treatment in this dataset. This is an OBSERVATIONAL")
    lines.append("  analysis and cannot prove causality. The result must NOT be read as")
    lines.append('  "support causes passing". Any difference may reflect who was selected')
    lines.append("  for support, not an effect of support itself.")
    lines.append("")
    lines.append("Treatment and outcome")
    lines.append("  Treatment: schoolsup (extra educational support).")
    lines.append(f"  Group sizes: schoolsup = yes: {n_t} students; schoolsup = no: {n_c} students.")
    lines.append("  Outcome: pass = (G3 >= 10). G3 was NEVER used as a feature (it defines the outcome).")
    lines.append(f"  Raw pass rate, schoolsup = yes: {rate_t * 100:.1f}% ({int(y[t == 1].sum())}/{n_t})")
    lines.append(f"  Raw pass rate, schoolsup = no:  {rate_c * 100:.1f}% ({int(y[t == 0].sum())}/{n_c})")
    lines.append(f"  Raw pass-rate difference (yes - no): {raw_diff_pp:+.2f} percentage points")
    lines.append("")
    lines.append("Confounders used for the propensity score (stated list)")
    lines.append("  school, sex, age, Medu, Fedu, studytime, failures, G1")
    lines.append("  Justification: each plausibly affects BOTH receiving extra support and passing.")
    lines.append("  G1 is a prior (first-period) grade, so it is pre-outcome and is a legitimate")
    lines.append("  confounder here; G2/G3 were excluded because they are post-treatment / outcome.")
    lines.append("  Categorical confounders (school, sex) were one-hot encoded; numeric confounders")
    lines.append("  were standardised (StandardScaler). Model: LogisticRegression (max_iter=1000).")
    lines.append(f"  Propensity-score range: {ps.min():.3f} to {ps.max():.3f};")
    lines.append(f"    treated range {ps[t == 1].min():.3f} to {ps[t == 1].max():.3f},")
    lines.append(f"    control range {ps[t == 0].min():.3f} to {ps[t == 0].max():.3f}.")
    lines.append("  Propensity scores were clipped to [0.01, 0.99] before IPW to avoid extreme weights.")
    lines.append("")
    lines.append("Method 1 — IPW (inverse-probability weighting)")
    lines.append(f"  IPW ATE on pass: {ate_ipw * 100:+.2f} percentage points")
    lines.append(f"  Standard error: bootstrap, {BOOTSTRAP_N} resamples, seed {BOOTSTRAP_SEED}, "
                 f"full pipeline refit per resample: {se_ipw * 100:.2f} percentage points")
    lines.append(f"  Bootstrap 95% percentile interval: [{ci_lo * 100:+.2f}, {ci_hi * 100:+.2f}] percentage points")
    lines.append(f"  (based on {len(boot)} successful bootstrap resamples)")
    lines.append("")
    lines.append("Method 2 — Nearest-neighbour propensity matching")
    lines.append(f"  Greedy matching on the propensity logit, caliper = {CALIPER_SD} SD of the logit")
    lines.append(f"  (logit SD = {logit_sd:.3f}, caliper = {caliper:.3f} on the logit scale), without replacement.")
    lines.append(f"  Treated units matched: {n_matched} of {n_t} ({n_matched / n_t * 100:.1f}%)")
    if pairs:
        lines.append(f"  Matched pass rate, treated: {matched_rate_t * 100:.1f}%")
        lines.append(f"  Matched pass rate, matched controls: {matched_rate_c * 100:.1f}%")
    lines.append(f"  Matched ATE on pass: {matched_ate * 100:+.2f} percentage points")
    if n_matched < n_t:
        lines.append(f"  Note: {n_t - n_matched} treated unit(s) could not be matched within the caliper,")
        lines.append("  so the matched ATE describes the matched subset only, not all treated students.")
    lines.append("")
    lines.append("Covariate balance — standardised mean differences (SMD)")
    lines.append("  |SMD| below ~0.1 is a common rough guide for acceptable balance.")
    for _, r in balance.iterrows():
        lines.append(f"  {r['variable']}: before {r['smd_before']:+.3f} | after IPW {r['smd_ipw']:+.3f} | "
                     f"after matching {r['smd_matched']:+.3f}")
    improved_w = bool((balance['smd_ipw'].abs() < balance['smd_before'].abs()).all())
    improved_m = bool((balance['smd_matched'].abs() < balance['smd_before'].abs()).all())
    lines.append(f"  Balance improved on all 4 checked variables after IPW weighting: {improved_w}")
    lines.append(f"  Balance improved on all 4 checked variables after matching: {improved_m}")
    lines.append("")
    lines.append("Interpretation — why this cannot be read causally")
    lines.append("  Ignorability (no unmeasured confounding) is assumed by both IPW and matching:")
    lines.append("  conditional on the stated confounders, receiving support must be as good as")
    lines.append("  random. That assumption cannot be verified from this file.")
    lines.append("  Unmeasured confounders plausibly remain, e.g. student motivation, teacher")
    lines.append("  judgement about who needs help, prior difficulties not captured by G1/failures,")
    lines.append("  and family circumstances beyond parental education. Support is typically GIVEN")
    lines.append("  to students who are struggling, so selection bias can mask or mimic an effect.")
    lines.append("  The raw gap, the IPW estimate and the matched estimate therefore describe")
    lines.append("  adjusted ASSOCIATIONS under strong, unverifiable assumptions — not proof that")
    lines.append('  support causes passing. Do not use these numbers to claim a causal effect.')
    lines.append("")
    lines.append("Files: results/results.txt (this file); visualizations/smd_before_after.png;")
    lines.append("  visualizations/propensity_overlap.png; main.py (reproducible, seed 42).")
    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "results.txt").write_text(text)
    print(text)
    print("Balance table:")
    print(balance.to_string(index=False))


if __name__ == "__main__":
    main()
