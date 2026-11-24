# Master Report — ML Lab Projects (all 20 labs)

## 1. Overview
This submission implements **all 20 Machine Learning lab problems** from the course PDF (`LAB_Projects_ML.pdf`), as **20 separate projects** sharing **one common dataset**. Every number in this report was produced by actually running the project's `main.py`; no metric was invented. Each project folder contains its own `main.py`, `README.md`, `results/results.txt`, and `visualizations/*.png`.

The first 15 labs fit the dataset directly. The last 5 (Labs 7, 9, 11, 13, 20) do **not** meet their printed conditions on this dataset, and are implemented as **honest, labelled adaptations** — each project's README and results state the caveat in plain words, and this report repeats it in §2 and §4. Nothing is presented as meeting a condition it does not meet.

## 2. The 20 labs and the selection
| Lab | Title | Selected? | Reason if not selected |
|---|---|---|---|
| 1 | Data Leakage | Yes — project_01 | |
| 2 | Missing Data | Yes — project_02 | |
| 3 | Distribution Shift | Yes — project_03 | |
| 4 | Explainability vs Performance | Yes — project_04 | |
| 5 | Custom Cross-Validation | Yes — project_05 | |
| 6 | Bias & Fairness | Yes — project_06 | |
| 7 | High-Dimensional Feature Selection (10,000+ features / 500 samples) | Yes — project_16 | ADAPTATION: original has 32 features, not 10,000+. Expanded to p = 1,769 by degree-2 polynomial expansion (p > n, but explicitly not 10,000). |
| 8 | Anomaly Detection | Yes — project_07 | |
| 9 | Time-Series Forecasting with Concept Drift | Yes — project_17 | ADAPTATION: no date column exists, so file order is a pseudo-time simulation only, not real seasonal forecasting. |
| 10 | Class Imbalance | Yes — project_08 | (Assumption caveat: see §4.) |
| 11 | Text Log Feature Engineering | Yes — project_18 | ADAPTATION: no raw logs exist; text is constructed from the structured fields, demonstrating the technique, not real log mining. |
| 12 | Calibration | Yes — project_09 | |
| 13 | Causal Inference | Yes — project_19 | ADAPTATION: no randomised treatment; observational IPW/matching analysis only, cannot prove causality. |
| 14 | Hyperparameter Search | Yes — project_10 | |
| 15 | Noise Robustness | Yes — project_11 | |
| 16 | Ensemble Diversity | Yes — project_12 | |
| 17 | Dimensionality Reduction | Yes — project_13 | |
| 18 | Streaming | Yes — project_14 | (File-order simulation only; see §4.) |
| 19 | Error Analysis | Yes — project_15 | |
| 20 | End-to-End System Design | Yes — project_20 | Working pipeline (validation, training, saved model, prediction demo, drift check) plus system-design documentation, not just a document. |

## 3. The shared dataset
- **Dataset:** UCI Student Performance, dataset 320 — https://archive.ics.uci.edu/dataset/320/student+performance
- **File used:** `student-por.csv` (Portuguese language course), saved untouched, semicolon-delimited, at `shared_dataset/dataset.csv`. UCI ships a second file, `student-mat.csv` (395 Math students), which is **deliberately not used**: it is smaller, and UCI notes 382 students appear in both files, so combining them would double-count students.
- **Domain:** education (Portuguese secondary-school students) — not banking or finance.
- **Size:** 649 rows × 33 columns, 0 missing values.
- **Target (derived, stated in every project):** `pass = (G3 >= 10)` — final grade at least 10 out of 20. Pass 549 (84.6%), fail 100 (15.4%).
- **Features (32, G3 excluded):** school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2.
- **Key facts:** school GP 423 students, pass 92.4%; school MS 226 students, pass 69.9%. Sex F 383, pass 86.9%; M 266, pass 81.2%. Grade correlations with G3: G1 0.826, G2 0.919. G3 mean 11.91, range 0–19.
- **Common setup:** stratified 75/25 train/test, seed 42 — train 486, test 163 (sklearn rounds 25% of 649 up from 162.25). Preprocessing: median/most-frequent imputation, StandardScaler for numerics, OneHotEncoder for categoricals, fit on train only, inside a Pipeline. G3 is never a feature.

## 4. Limitations and honest caveats (read before judging any number)
- **Small sample.** 649 rows means test sets of 163 rows and calibration sets of 98 rows; small differences (e.g. 0.001 in CV ROC-AUC) are noise, and several projects say so explicitly.
- **Lab 10's printed assumption (<2% target) is NOT met.** The fail rate is 15.4%. Project_08 states this plainly and still compares the imbalance methods; the baseline wins, partly because the imbalance is only moderate.
- **No date column.** No project claims a verified chronological split or a real time series. Project_14's streaming order is file order, labelled as a simulation only; project_03's shift is across schools, not across time.
- **No missing values in the original file.** Project_02 first proves this (all 33 columns 649/649 complete), then introduces labelled MCAR/MAR/MNAR missingness as simulations — never presented as original data.
- **G1/G2 timing.** Earlier-period grades are legitimate for end-of-term prediction but are future information for an early-warning model. Project_01 makes this distinction the whole point of the leakage lab.
- **Anomaly labels do not exist.** Project_07's validation is proxy-only (fail rate, mean G3, absences among flagged rows), explicitly not accuracy.
- **Calibration set is tiny (98 rows).** In project_09 no calibrated variant beat the best uncalibrated model; the recommendation is among calibrated variants only, and the report says so.
- **Labs 7, 9, 11, 13 adaptations (projects_16–19).** Lab 7 reaches p = 1,769, not 10,000; Lab 9 has no real timestamps; Lab 11 has no real logs; Lab 13 has no randomisation, and its IPW estimate (+17.15 pp, 95% interval [−10.29, +40.64] pp, only 68 treated students) is unstable and includes zero, while the matched estimate is +3.12 pp — neither may be read as "support causes passing". These caveats are in each project's README and results, not buried here.

## 5. Folder layout
```
ML_Lab_Projects_v4/
├── master_report.md
├── requirements.txt
├── dataset_analysis.py
├── shared_dataset/
│   ├── dataset.csv            (UCI student-por.csv, untouched)
│   ├── dataset_information.md
│   └── student.txt            (UCI attribute documentation)
└── project_01/ … project_20/
    ├── main.py
    ├── README.md
    ├── results/results.txt
    └── visualizations/*.png
    (project_20 also ships results/model.joblib, the trained pipeline)
```

## 6. Per-project summaries (actual measured results)

### project_01 — Lab 1: Data Leakage
Three RandomForest models, same split: (a) leaky model including G3 as a feature — ROC-AUC **1.0000**, accuracy 1.0000 (it reads the answer); (b) end-of-term model with G1+G2 — ROC-AUC **0.9432**, accuracy 0.9141; (c) early-warning model with no G1/G2/G3 — ROC-AUC **0.7348**, accuracy 0.8282. Leakage inflation (a − c) = **0.2652** ROC-AUC. Recommended pipeline for early warning is (c).

### project_02 — Lab 2: Missing Data
Real file: **0 missing values**. Labelled simulations, RandomForest test ROC-AUC, median/mode vs KNN(k=5): MCAR (4,165 cells, 20.1%) **0.9319 vs 0.9280**; MAR (216 cells) **0.9451 vs 0.9446**; MNAR (148 cells) **0.9442 vs 0.9423**. Simple median/mode imputation matched KNN here, largely because G1/G2 carry most of the signal and were never masked in MAR/MNAR.

### project_03 — Lab 3: Distribution Shift
Cross-school RandomForest ROC-AUC, all features vs stable-only (24 of 31 features with KS/proportion-difference < 0.20; G1 KS 0.321 and G2 KS 0.277 dropped): GP→MS **0.9453 vs 0.6851 (−0.2602)**; MS→GP **0.9734 vs 0.7712 (−0.2022)**. Honest verdict: stable-only retraining did **not** help in either direction — dropping the shifted grades removed the strongest signals.

### project_04 — Lab 4: Explainability vs Performance
LogisticRegression ROC-AUC **0.9420** (accuracy 0.9202); depth-4 DecisionTree **0.7797** (0.8896); RandomForest 200 trees **0.9432** (0.9141). Permutation importance top: G2 0.1262, G1 0.0644. SHAP (first 100 test rows) top: G1 0.0793, G2 0.0744. Both methods agree earlier grades dominate; logistic regression essentially matches the forest here, so it is the better interpretable choice on this dataset.

### project_05 — Lab 5: Custom Cross-Validation
Manual loops only (no `cross_val_score`), LogisticRegression, mean ROC-AUC ± std: shuffled 5-fold **0.9549 ± 0.0150**; stratified 5-fold **0.9529 ± 0.0127**; LeaveOneGroupOut by school (2 folds) **0.9295 ± 0.0079**. The grouped score is the honest estimate for deployment to an unseen school; shuffled folds mix both schools and are optimistic for that deployment.

### project_06 — Lab 6: Bias & Fairness
Sensitive attribute sex (real pass rates F 86.9% vs M 81.2%). Baseline LogisticRegression ROC-AUC **0.9420**, sex demographic parity difference 0.0134, equal-opportunity difference 0.0079. Manual reweighing (train-only): ROC-AUC **0.9307**, sex DPD 0.0098, EOD 0.0013 — but school gaps worsened sharply (DPD 0.2664→0.4104, EOD 0.1014→0.3090). Verdict: reweighing did not help fairness overall; it only rebalanced the attribute it was weighted on.

### project_07 — Lab 8: Anomaly Detection
Unsupervised, full 649 rows, contamination 0.10: IsolationForest flagged 65, LOF flagged 65, both flagged 35, Jaccard overlap **0.3684**. Proxy-only validation: rows flagged by both had fail rate **48.6%**, mean G3 9.0, mean absences 6.2, vs unflagged-by-either 12.1%, 12.23, 3.24. This is a proxy description, not accuracy — no ground-truth anomalies exist.

### project_08 — Lab 10: Class Imbalance
Fail-class F1 (test): baseline **0.7451** (precision 0.7308, recall 0.7600, ROC-AUC 0.9420); class_weight balanced 0.6562; SMOTE (train only) 0.6780; random undersampling 0.5479; threshold tuning (threshold 0.73, tuned on train out-of-fold only) 0.6780. Recommendation: **baseline** — the highest fail-class F1. The lab's <2% assumption is not met (fail rate 15.4%), which is why the gains from resampling are modest.

### project_09 — Lab 12: Calibration
Split train 388 / calibration 98 / test 163. Manual 10-bin ECE. Best uncalibrated: HistGradientBoosting (Brier **0.0599**, ROC-AUC 0.9614, ECE 0.0452). Recommended among calibrated variants: HistGradientBoosting + sigmoid (Brier 0.0669). Caveat recorded in the project: no calibrated variant beat the best uncalibrated model, likely because only 98 calibration rows were available (sklearn 1.9 removed `CalibratedClassifierCV(cv='prefit')`, so sigmoid/isotonic calibration was implemented explicitly on the calibration set only).

### project_10 — Lab 14: Hyperparameter Search
RandomForest on the full 649 rows, 3-fold CV on train, 12 configs each: grid **0.9702** (14.5 s); random **0.9698** (14.5 s); manual Bayesian GP + Expected Improvement **0.9708** (11.2 s; max_depth 20, min_samples_leaf 7, n_estimators 200). Recommendation: Bayesian, with the caveat that all three are effectively tied (within 0.001) on 486 training rows.

### project_11 — Lab 15: Noise Robustness
Clean RandomForest baseline: ROC-AUC **0.9432**, F1 0.9496, accuracy 0.9141. Labelled simulations: 10% label flip → 0.9429; 20% label flip → **0.8967**; Gaussian noise on scaled numerics → 0.9028; 10% `Mjob` corruption → 0.9432 (no change). Fixes on the 20%-flip scenario: out-of-fold-cleaned retrain (kept 354/486 rows) → **0.9422** (recovered); max_depth=8 regularised model → 0.9342 (recovered).

### project_12 — Lab 16: Ensemble Diversity
Single models, test ROC-AUC: LogisticRegression 0.9420, RandomForest 0.9432, HistGradientBoosting **0.9597**, KNN 0.7996. Soft-voting all four: 0.9403. Most redundant by mean Q-statistic: RandomForest (0.9733); pruned ensemble without it: 0.9368. Honest verdict: the best single model won; diversity did not help here.

### project_13 — Lab 17: Dimensionality Reduction
Full 649 rows, 58 preprocessed features. Pre-stated rule: recommend the method with highest 5-NN CV accuracy in 2-D, ties by trustworthiness. PCA: 5-NN accuracy 0.8536, trustworthiness 0.7578, silhouette 0.2380 (descriptive only). t-SNE: 0.8458, 0.9251, 0.1511. UMAP: **0.8551**, 0.8470, 0.1493. **Recommended: UMAP.**

### project_14 — Lab 18: Streaming Simulation
File order only (no timestamp exists), 7 batches (6×100 + 49), preprocessing fit on batch 0 only, scaled values clipped to [−10, 10]. (a) SGD prequential mean ROC-AUC **0.7962** over batches 1–6 (batch 1 was 0.5000 after only 100 rows). (b) Retraining trigger (ROC-AUC < 0.70): **0 triggers in 6 batches**, final-batch ROC-AUC 0.8803 — the trigger mechanism was implemented but never fired, so (b) is effectively the batch-0 model evaluated onward, stated as such.

### project_15 — Lab 19: Error Analysis
5-fold out-of-fold RandomForest on the full 649 rows: ROC-AUC **0.9669**, F1 0.9564, accuracy 0.9245. Worst slice (≥30 rows): students with ≥1 past failure, error rate **0.2100** (n=100), vs 0.0510 for students with no failures; school MS 0.1327, sex M 0.1053. One targeted fix (sample weight 2.0 for the worst-slice training rows): worst-slice error 0.2100 → **0.2000** (slight improvement), overall ROC-AUC 0.9695, accuracy unchanged at 0.9245.

### project_16 — Lab 7: High-Dimensional Feature Selection (ADAPTATION)
Original dataset: 32 features, not 10,000+ — stated condition **not met**. Degree-2 polynomial expansion of the 58 preprocessed features gives **p = 1,769** (p > n, but not 10,000). Same split: reference L2 logistic regression on 58 features — ROC-AUC 0.9420; L2 on all 1,769 features — 0.9377 (train ROC-AUC 1.0000, near-memorisation); L1 selection — **0.9664** with 22 non-zero features; SelectKBest k=50 — 0.9649; stability selection (20 bootstraps, ≥50%) — 0.9577 with 13 features. L1 and stability sets overlap fully on the 13 stability features (Jaccard 0.5909). Honest reading: selection recovered comparable-or-better test ROC-AUC with far fewer features, but bought no reliable gain over the simple 58-feature model.

### project_17 — Lab 9: Time-Series Forecasting with Concept Drift (SIMULATION)
No date column exists — file order is a **pseudo-time simulation only**. Seven batches (6×100 + 49). Batch-mean G3 forecast MAE: rolling RandomForest model **0.3326** vs previous-batch baseline 1.1677; pass-rate forecast error 0.0752 vs 0.1108; mean per-student G3 MAE rolling 0.9636 vs batch-0-only model 1.2781 (rolling better on 5 of 6 batches). Drift (KS on G1/G2, p < 0.05) flagged at boundaries 0→1, 2→3, 3→4, 4→5; largest shift 3→4 (mean G3 −1.99, pass rate −0.16).

### project_18 — Lab 11: Text Log Feature Engineering (ADAPTATION)
No raw logs exist — text is **constructed from structured fields** (actual field values only, G3 and the word "pass" excluded). Same split, re-run baseline: structured LogisticRegression ROC-AUC **0.9420**; TF-IDF (word n-grams 1–2) text-only **0.8916**; combined structured + TF-IDF **0.9455**. Top text terms by coefficient include positive `11` (+1.2477) and `g2 11` (+0.8566), negative `7` (−2.4040) and `g2 8` (−1.6398) — the text model rediscovers the grade signal in token form.

### project_19 — Lab 13: Causal Inference (OBSERVATIONAL ONLY)
No randomised treatment — this **cannot prove causality**. Treatment: schoolsup (extra educational support), yes 68 vs no 581 students. Raw pass rates 88.2% vs 84.2% → raw difference **+4.07 pp**. Confounders: school, sex, age, Medu, Fedu, studytime, failures, G1. IPW ATE **+17.15 pp**, bootstrap SE 14.39 pp, 95% interval [−10.29, +40.64] pp (includes zero — unstable with only 68 treated). Matched ATE (64/68 matched) **+3.12 pp**. Covariate balance did not become clean on all checked variables (e.g. G1 SMD −0.246 → −0.147 after matching). Verdict: adjusted associations under strong, unverifiable assumptions — not proof that support causes passing.

### project_20 — Lab 20: End-to-End System Design
Working pipeline, not just a document: validation passed (649 rows, 0 missing, G3 range 0–19, pass rate 84.59%); model HistGradientBoosting re-fit here; test ROC-AUC **0.9597**, accuracy 0.9080, fail-class F1 0.6809; pipeline saved to `results/model.joblib` and reload-verified; predictions demonstrated on 3 test rows; drift check (KS train vs test: G1 0.0632, G2 0.0404, absences 0.0434, mean 0.0490) — **no retraining trigger** under the stated rule (any KS > 0.20 or mean KS > 0.10). README documents the full architecture: ingest → validate → preprocess → train → evaluate → serve → monitor → retrain, with deployment notes, monitoring metrics, and failure modes.

## 7. Cross-project comparisons
- **Best single-model ROC-AUC on the common split:** HistGradientBoosting 0.9597–0.9614 (projects_09/12), ahead of RandomForest 0.9432 and LogisticRegression 0.9420. The ranking is consistent across projects that use the same split.
- **The dominant signal everywhere is G1/G2.** Removing them (early-warning model, project_01) drops ROC-AUC from 0.9432 to 0.7348; removing them again (stable-only, project_03) drops cross-school ROC-AUC by 0.20–0.26. Any "student performance" claim that silently drops or keeps these grades is answering a different question — the projects state which question they answer.
- **Simpler methods often tie the complex ones here.** Logistic regression matches the forest (project_04), the baseline beats SMOTE and class-weighting (project_08), and no ensemble beats the best single model (project_12). On 649 rows with a strong grade signal, complexity buys little — that is a finding, not a failure.
- **Several "fixes" honestly failed.** Stable-only retraining (project_03), sex reweighing for overall fairness (project_06), and calibration on 98 rows (project_09) all made things worse or no better, and each project reports that instead of claiming a win.
- **The school gap is the dataset's biggest real-world problem.** GP passes at 92.4% vs MS at 69.9%; models make roughly three times as many errors on MS students (project_15: 0.1327 vs 0.0449) and on students with past failures (0.2100). A model deployed to a new school should be judged by the grouped score (0.9295, project_05), not the shuffled score (0.9549).

## 8. Observations
1. Leakage is a timing question, not just a column question: the same G1/G2 columns are legitimate end-of-term and leaky early-warning.
2. Distribution shift here is institutional (two schools with different student mixes and pass rates), not temporal — and the "robust" fix of dropping shifted features backfires when the shifted features are also the predictive ones.
3. Fairness interventions optimise the attribute they are weighted on and can worsen other slices; fairness must be checked across slices, not on one gap.
4. With a small, clean dataset, the honest contributions of these labs are methodological (correct splits, correct grouping, correct calibration protocol, honest proxy validation) rather than large accuracy gains.

## 9. How to install and run
```bash
pip install -r requirements.txt
python dataset_analysis.py          # reproduces the dataset facts in §3
cd project_01 && python main.py     # each project re-creates its own results/ and visualizations/
```
Each `main.py` loads only `../shared_dataset/dataset.csv` (relative to its own file), so projects can be run independently in any order. Re-running overwrites that project's `results/results.txt` and PNGs with fresh numbers.

## 10. Conclusion
All twenty labs are implemented end to end on a single, real, non-banking dataset (UCI Student Performance, Portuguese course, 649 students), with every reported number produced by running code, every simulation and adaptation labelled as such, and every failed fix reported as a failure. The five labs whose printed conditions this dataset does not meet (7, 9, 11, 13) are implemented as labelled adaptations, and Lab 20 as a working pipeline — none is presented as meeting a condition it does not meet. The strongest, most defensible results are the methodological ones: the leakage timing analysis (0.2652 ROC-AUC inflation), the school-grouped validation score (0.9295 vs shuffled 0.9549), and the error analysis showing where the model actually fails (students with past failures, 21.0% error).
