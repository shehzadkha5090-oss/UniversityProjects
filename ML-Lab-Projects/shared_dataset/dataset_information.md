# Shared Dataset — ML Lab Projects v4

## Name and source
- **Dataset:** UCI Student Performance (dataset 320)
- **Page:** https://archive.ics.uci.edu/dataset/320/student+performance
- **File used:** `student-por.csv` (Portuguese language course), saved untouched at `shared_dataset/dataset.csv` (semicolon-delimited, exactly as UCI publishes it).
- **Domain:** education — explicitly NOT banking/finance.
- UCI ships two files with the same 33 attributes: `student-mat.csv` (395 Math students) and `student-por.csv` (649 Portuguese students). Only the Portuguese file is used here: it is the larger file, and using both would double-count the 382 students who appear in both files (UCI's own note in `student.txt`). The Math file is deliberately NOT used.

## Size and target
- **649 rows × 33 columns.** No rows added, dropped, or invented; no missing values (0 NaNs in the original file).
- **Target (derived, stated plainly):** `pass = (G3 >= 10)`, i.e. final grade at least 10 out of 20.
  - Pass: 549 students (84.6%); Fail: 100 students (15.4%). The fail class is the minority.
  - G3 mean 11.91, std 3.23, range 0–19.
- Original `G3` is retained in the file; projects must NOT use `G3` as a feature when predicting `pass` (that would be target leakage — see project_01).

## Columns (32 features + G3)
- School/demographic: `school` (GP Gabriel Pereira 423 / MS Mousinho da Silveira 226), `sex` (F 383 / M 266), `age` (15–22), `address` (U/R), `famsize`, `Pstatus`.
- Family/parent: `Medu`, `Fedu` (0–4), `Mjob`, `Fjob`, `reason`, `guardian`.
- Study context: `traveltime`, `studytime`, `failures`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`.
- Wellbeing/social: `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health` (all 1–5 scales), `absences` (0–93, mean 3.66).
- Grades: `G1` first period, `G2` second period, `G3` final (0–20). Correlations with G3: G1 0.826, G2 0.919, G1–G2 0.865.

## Key facts projects must respect
- **Pass rate by school:** GP 92.4% vs MS 69.9% — a large real school gap; ideal for the shift and grouped-CV labs, and a warning against claiming a model trained on one school generalises to the other.
- **Pass rate by sex:** F 86.9% vs M 81.2% — the fairness lab's primary sensitive attribute is `sex` (F vs M).
- **G1/G2 timing:** G1 and G2 are grades from earlier periods in the same course. They are legitimate features for an end-of-term prediction, but they are FUTURE information for an early-warning model that must act at the start of the year — project_01 treats this timing distinction as its leakage question and reports both models honestly.
- **No date/time column exists** in this dataset (no school year, term date, or timestamp). So no project may claim a verified chronological split or a true calendar time series; shift and streaming work uses labelled school slices and file-order simulations respectively, and says so.
- **No missing values** in the original file: the missing-data lab (project_02) analyses the real completeness first, then introduces labelled MCAR/MAR/MNAR missingness as simulations, never as original data.
- **Small sample (649 rows):** results are noisier than on a large dataset. Projects use stratified splits (seed 42), small test sets are reported with their sizes, and no subsample larger than 649 is ever claimed. Hyperparameter search and dimensionality-reduction use the full 649 rows (or a stated smaller subsample), never a fictional 15,000.
- **Lab 10's printed assumption (<2% target) is NOT met:** the fail rate is 15.4%. Project_08 states this plainly instead of pretending otherwise.
- **Excluded labs (same reasons as before, re-checked against this dataset):** Lab 7 (needs 10,000+ features; this dataset has 32), Lab 9 (needs a continuous seasonal time series; no date column), Lab 11 (needs raw log text; none here), Lab 13 (no randomised/defensible treatment variable), Lab 20 (no standalone experiment beyond the master report/pipelines).
