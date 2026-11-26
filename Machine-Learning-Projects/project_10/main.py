"""Lab 14 Hyperparameter Search — UCI Student Performance (student-por.csv).
RandomForest on the FULL 649 rows (never a 15,000-row subsample — the dataset
has only 649 rows). Comparison on 3-fold CV on the TRAIN set only (train 486,
test 163, stratified 75/25 seed 42; test is not used for search).
Grid: small grid; Random: same budget of 12 configs; Bayesian: MANUAL
Gaussian-Process surrogate + Expected Improvement over 2 parameters,
max_depth (encoded: None=25, else its value, searched on [3, 25]) and
min_samples_leaf (searched on [1, 10], rounded to int), n_estimators fixed 200.
Encoding is stated here and in the README. CV score: ROC-AUC."""
import time, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parents[1]
OUT, VIS = BASE / "project_10" / "results", BASE / "project_10" / "visualizations"
OUT.mkdir(parents=True, exist_ok=True); VIS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "shared_dataset" / "dataset.csv", sep=";")
y = (df["G3"] >= 10).astype(int); X = df.drop(columns=["G3"])
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
cv = StratifiedKFold(3, shuffle=True, random_state=42)
def pipe(**kw): return Pipeline([("pre", pre), ("clf", RandomForestClassifier(random_state=42, **kw))])
def cv_score(**kw): return float(cross_val_score(pipe(**kw), Xtr, ytr, cv=cv, scoring="roc_auc").mean())

t0 = time.time(); grid = GridSearchCV(pipe(), {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 5, 10], "clf__min_samples_leaf": [1, 2]}, cv=cv, scoring="roc_auc")
grid.fit(Xtr, ytr); t_grid = time.time() - t0
t0 = time.time(); rnd = RandomizedSearchCV(pipe(), {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [None, 5, 10, 15], "clf__min_samples_leaf": [1, 2, 4], "clf__max_features": ["sqrt", None]}, n_iter=12, cv=cv, scoring="roc_auc", random_state=42)
rnd.fit(Xtr, ytr); t_rnd = time.time() - t0

# Manual Bayesian: 5 random starts + 7 EI iterations = 12 CV evaluations
rng = np.random.default_rng(42); t0 = time.time()
Xb, yb = [], []
for _ in range(5):
    d, leaf = float(rng.uniform(3, 25)), int(rng.integers(1, 11)); s = cv_score(n_estimators=200, max_depth=None if d >= 24.5 else int(round(d)), min_samples_leaf=leaf)
    Xb.append([d, leaf]); yb.append(s)
for _ in range(7):
    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True, random_state=42); gp.fit(np.array(Xb), np.array(yb))
    cand = np.column_stack([rng.uniform(3, 25, 500), rng.integers(1, 11, 500)])
    mu, sd = gp.predict(cand, return_std=True); best = max(yb)
    z = (mu - best) / np.maximum(sd, 1e-9); ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z); ei[sd == 0] = 0
    d, leaf = cand[int(np.argmax(ei))]; leaf = int(leaf)
    s = cv_score(n_estimators=200, max_depth=None if d >= 24.5 else int(round(d)), min_samples_leaf=leaf)
    Xb.append([float(d), leaf]); yb.append(s)
t_bayes = time.time() - t0
bi = int(np.argmax(yb)); bd, bleaf = Xb[bi]
bayes_params = dict(n_estimators=200, max_depth=None if bd >= 24.5 else int(round(bd)), min_samples_leaf=int(bleaf))

lines = ["Lab 14 Hyperparameter Search — UCI Student Performance (student-por.csv)",
 "RandomForest on the FULL 649 rows (no subsample claim). Search: 3-fold CV on train only.",
 f"Split: stratified 75/25 seed 42 — train {len(ytr)}, test {len(yte)} (test not used for search).",
 "Bayesian encoding: max_depth searched on [3, 25] with 25 mapped to None; min_samples_leaf on [1, 10] (int); n_estimators fixed 200.",
 f"Grid search: best CV ROC-AUC={grid.best_score_:.4f}, best params={grid.best_params_}, time={t_grid:.1f}s, configs=12",
 f"Random search: best CV ROC-AUC={rnd.best_score_:.4f}, best params={rnd.best_params_}, time={t_rnd:.1f}s, configs=12",
 f"Manual Bayesian (GP + Expected Improvement): best CV ROC-AUC={max(yb):.4f}, best params={bayes_params}, time={t_bayes:.1f}s, evaluations=12"]
scores = {"Grid": grid.best_score_, "Random": rnd.best_score_, "Bayesian": max(yb)}
rec = max(scores, key=scores.get)
lines.append(f"Recommendation: {rec} search — highest 3-fold CV ROC-AUC ({scores[rec]:.4f}) at equal budget (12 configs);")
lines.append("  if two methods tie within noise, prefer the simpler/faster one, since 486 train rows make small CV differences unreliable.")
(OUT / "results.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(list(scores), list(scores.values())); ax.set_ylabel("Best 3-fold CV ROC-AUC"); ax.set_title("Lab 14: Best CV ROC-AUC by search method")
fig.tight_layout(); fig.savefig(VIS / "search_comparison.png", dpi=150)
print("Saved search_comparison.png")
