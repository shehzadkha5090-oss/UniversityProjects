# Lab 13 — Causal Inference (project_19)

## Caveat — read first

There is NO randomised treatment in this dataset. This is an OBSERVATIONAL
analysis and cannot prove causality. The result must NOT be read as
"support causes passing". Any difference may reflect who was selected for
support, not an effect of support itself.

## Dataset

UCI Student Performance, `student-por.csv` via
`../shared_dataset/dataset.csv` — 649 rows × 33 columns, semicolon-delimited.
No other dataset was used. `student-mat.csv` was not used.

## Treatment and outcome

- **Treatment:** `schoolsup` (extra educational support, yes/no).
- **Group sizes (verified from the file):** schoolsup = yes: **68 students**;
  schoolsup = no: **581 students**.
- **Outcome:** pass = (G3 ≥ 10). G3 was NEVER used as a feature — it defines
  the outcome.

### Raw results (reported first, before any adjustment)

- Raw pass rate, schoolsup = yes: **88.2% (60/68)**
- Raw pass rate, schoolsup = no: **84.2% (489/581)**
- Raw pass-rate difference (yes − no): **+4.07 percentage points**

## Methods

### 1. Propensity-score estimation

LogisticRegression on the stated confounders — **school, sex, age, Medu,
Fedu, studytime, failures, G1** — with one-hot encoding for the categorical
confounders (school, sex) and StandardScaler for the numeric confounders.

Justification: each of these plausibly affects BOTH receiving extra support
and passing. G1 is a prior (first-period) grade, so it is pre-outcome and is
a legitimate confounder here; G2/G3 were excluded because they are
post-treatment / outcome variables. Propensity scores were clipped to
[0.01, 0.99] before IPW to avoid extreme weights.

Propensity-score range: 0.010 to 0.617; treated range 0.010 to 0.617;
control range 0.010 to 0.555.

### 2. IPW (inverse-probability weighting)

- **IPW ATE on pass: +17.15 percentage points**
- **Standard error: bootstrap, 200 resamples, seed 42, full pipeline refit
  per resample: 14.39 percentage points**
- **Bootstrap 95% percentile interval: [−10.29, +40.64] percentage points**
  (based on 200 successful bootstrap resamples)

The interval is very wide and includes zero and negative values, so the IPW
estimate is imprecise and does not establish an effect. It is much larger
than the raw (+4.07 pp) and matched (+3.12 pp) estimates, which is a warning
sign: with only 68 treated students, IPW is sensitive to a few low-propensity
weights.

### 3. Nearest-neighbour propensity matching

Greedy matching on the propensity logit, caliper = 0.2 SD of the logit
(logit SD = 1.087, caliper = 0.217 on the logit scale), without replacement.

- **Treated units matched: 64 of 68 (94.1%)** — 4 treated units could not be
  matched within the caliper, so the matched ATE describes the matched
  subset only, not all treated students.
- Matched pass rate, treated: 89.1%
- Matched pass rate, matched controls: 85.9%
- **Matched ATE on pass: +3.12 percentage points**

### 4. Covariate balance (standardised mean differences, SMD)

|SMD| below ~0.1 is a common rough guide for acceptable balance.

| Variable | Before | After IPW | After matching |
|---|---|---|---|
| failures | −0.002 | +0.140 | +0.135 |
| G1 | −0.246 | +0.201 | −0.147 |
| Medu | −0.073 | −0.073 | +0.000 |
| school (GP=1) | +0.440 | −0.273 | −0.082 |

Balance improved on all 4 checked variables after IPW weighting: **False**.
Balance improved on all 4 checked variables after matching: **False**.

Stated plainly: balance did **not** become clean. Matching fixed school
(+0.440 → −0.082) and Medu (−0.073 → +0.000), but G1 and failures remained
above the 0.1 guide after both IPW and matching, and failures was already
essentially balanced before adjustment (−0.002), so weighting/matching made
it slightly worse. The adjusted estimates therefore still carry residual
imbalance and should not be over-read.

## Why this cannot be read causally

Both IPW and matching assume ignorability (no unmeasured confounding):
conditional on the stated confounders, receiving support must be as good as
random. That assumption cannot be verified from this file. Unmeasured
confounders plausibly remain, e.g. student motivation, teacher judgement
about who needs help, prior difficulties not captured by G1/failures, and
family circumstances beyond parental education. Support is typically GIVEN
to students who are struggling, so selection bias can mask or mimic an
effect. The raw gap, the IPW estimate and the matched estimate therefore
describe adjusted ASSOCIATIONS under strong, unverifiable assumptions —
not proof that support causes passing.

## Reproduce

```bash
~/workspace/ML_Lab_Projects_v4/.venv/bin/python main.py
```

## Files

- `main.py` — self-contained; loads the dataset via
  `Path(__file__).resolve().parents[1] / 'shared_dataset' / 'dataset.csv'`
- `results/results.txt` — full results (numbers in this README are copied
  from it, not invented)
- `visualizations/smd_before_after.png` — SMD before vs after adjustment
- `visualizations/propensity_overlap.png` — propensity-score overlap
