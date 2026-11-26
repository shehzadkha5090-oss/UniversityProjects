"""Lab 16 - Ensemble Diversity (UCI Student Performance, student-por.csv).
Target: pass = (G3 >= 10). G3 never a feature. Stratified 75/25, seed 42.
"""
from pathlib import Path
import numpy as np, pandas as pd, itertools
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "shared_dataset" / "dataset.csv", sep=";")
y = (df["G3"] >= 10).astype(int); X = df.drop(columns=["G3"])
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
def pipe(model):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    return Pipeline([("pre", pre), ("m", model)])
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
lines = ["Dataset: UCI Student Performance (student-por.csv), 649 rows",
         "Target: pass = (G3 >= 10); Pass 549 (84.6%), Fail 100 (15.4%); G3 never a feature",
         f"Split: stratified 75/25, seed 42; train {len(Xtr)}, test {len(Xte)}"]
preds, probas, aucs = {}, {}, {}
for name, mdl in models.items():
    p = pipe(mdl); p.fit(Xtr, ytr)
    probas[name] = p.predict_proba(Xte)[:, 1]; preds[name] = (probas[name] >= 0.5).astype(int)
    aucs[name] = roc_auc_score(yte, probas[name])
    lines.append(f"{name}: test ROC-AUC={aucs[name]:.4f}")
# Soft voting of all four (fresh pipelines)
vote = VotingClassifier([(n, pipe(m)) for n, m in models.items()], voting="soft")
vote.fit(Xtr, ytr); va = roc_auc_score(yte, vote.predict_proba(Xte)[:, 1])
lines.append(f"Soft-voting ensemble (all 4): test ROC-AUC={va:.4f}")
# Diversity: pairwise disagreement + Q-statistic (own code)
def q_stat(a_correct, b_correct):
    n11 = int(((a_correct == 1) & (b_correct == 1)).sum()); n00 = int(((a_correct == 0) & (b_correct == 0)).sum())
    n10 = int(((a_correct == 1) & (b_correct == 0)).sum()); n01 = int(((a_correct == 0) & (b_correct == 1)).sum())
    den = n11 * n00 + n10 * n01
    return (n11 * n00 - n10 * n01) / den if den else 0.0
correct = {n: (preds[n] == yte.values).astype(int) for n in preds}
mean_q = {}
for a, bname in itertools.combinations(models, 2):
    dis = float((preds[a] != preds[bname]).mean())
    q = q_stat(correct[a], correct[bname])
    lines.append(f"Diversity {a} vs {bname}: disagreement rate={dis:.4f}, Q-statistic={q:.4f}")
for n in models:
    qs = [q_stat(correct[n], correct[o]) for o in models if o != n]
    mean_q[n] = float(np.mean(qs))
    lines.append(f"{n}: mean Q vs others={mean_q[n]:.4f} (higher = more redundant)")
pruned = max(mean_q, key=mean_q.get)
lines.append(f"Pruned model (highest mean Q, most redundant): {pruned}")
remaining = [(n, pipe(m)) for n, m in models.items() if n != pruned]
vote2 = VotingClassifier(remaining, voting="soft"); vote2.fit(Xtr, ytr)
va2 = roc_auc_score(yte, vote2.predict_proba(Xte)[:, 1])
lines.append(f"Pruned soft-voting ensemble (without {pruned}): test ROC-AUC={va2:.4f}")
best_single = max(aucs, key=aucs.get)
lines.append(f"Best single model: {best_single} ROC-AUC={aucs[best_single]:.4f}")
if va >= aucs[best_single] and va2 >= aucs[best_single]:
    verdict = "Diversity helped: both ensembles beat or matched the best single model."
elif va2 > va:
    verdict = "Pruning helped: the pruned ensemble beat the full ensemble, but the verdict vs the best single model is as reported above."
else:
    verdict = "The best single model won: neither ensemble beat it, so diversity did not help on this test set."
lines.append("Discussion: " + verdict)
(OUT / "results").mkdir(exist_ok=True); (OUT / "visualizations").mkdir(exist_ok=True)
(OUT / "results" / "results.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
names = list(aucs) + ["Voting-all4", f"Voting-no-{pruned}"]
vals = list(aucs.values()) + [va, va2]
plt.figure(figsize=(9, 4)); plt.bar(names, vals); plt.ylabel("Test ROC-AUC"); plt.title("Ensemble diversity: single models vs voting ensembles")
plt.xticks(rotation=20, ha="right"); plt.tight_layout()
plt.savefig(OUT / "visualizations" / "ensemble_diversity.png", dpi=150); plt.close()
