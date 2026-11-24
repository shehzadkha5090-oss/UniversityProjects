"""Lab 17 - Dimensionality Reduction (UCI Student Performance, student-por.csv).
FULL 649 rows, preprocessed (median/most-frequent impute, StandardScaler, OneHot).
Target pass = (G3 >= 10) used ONLY for descriptive silhouette / colouring, never as input.
PRE-STATED RECOMMENDATION RULE (fixed before seeing results):
  Recommend the method with the highest 5-NN cross-validated accuracy in the 2-D
  space; ties broken by higher trustworthiness. This rule is applied below as written.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import umap
    HAS_UMAP = True
except Exception as e:
    HAS_UMAP = False; UMAP_ERR = str(e)

BASE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "shared_dataset" / "dataset.csv", sep=";")
y = (df["G3"] >= 10).astype(int).values; X = df.drop(columns=["G3"])
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
Xa = pre.fit_transform(X)
if hasattr(Xa, "toarray"): Xa = Xa.toarray()
lines = ["Dataset: UCI Student Performance (student-por.csv), FULL 649 rows (no train/test split here)",
         "Target: pass = (G3 >= 10); Pass 549 (84.6%), Fail 100 (15.4%); G3 never an input feature",
         "PRE-STATED RECOMMENDATION RULE: recommend the method with the highest 5-NN CV accuracy in 2-D; ties broken by trustworthiness.",
         f"Preprocessed feature matrix: {Xa.shape}"]
emb = {}
emb["PCA"] = PCA(n_components=2, random_state=42).fit_transform(Xa)
emb["t-SNE"] = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(Xa)
if HAS_UMAP:
    emb["UMAP"] = umap.UMAP(n_components=2, n_neighbors=15, random_state=42).fit_transform(Xa)
else:
    lines.append(f"UMAP unavailable in this environment ({UMAP_ERR}); UMAP omitted, rule applied to PCA and t-SNE only.")
results = {}
for name, E in emb.items():
    sil = silhouette_score(E, y)  # descriptive only, on pass/fail labels
    knn = cross_val_score(KNeighborsClassifier(5), E, y, cv=5).mean()
    tru = trustworthiness(Xa, E, n_neighbors=5)
    results[name] = (knn, tru)
    lines.append(f"{name}: silhouette (descriptive only)={sil:.4f}, 5-NN CV accuracy={knn:.4f}, trustworthiness={tru:.4f}")
    plt.figure(figsize=(6, 5)); plt.scatter(E[y == 0, 0], E[y == 0, 1], s=12, label="Fail (G3<10)")
    plt.scatter(E[y == 1, 0], E[y == 1, 1], s=12, label="Pass (G3>=10)")
    plt.legend(); plt.title(f"{name} 2-D embedding coloured by pass/fail"); plt.tight_layout()
    (OUT / "visualizations").mkdir(exist_ok=True)
    plt.savefig(OUT / "visualizations" / f"{name.lower().replace('-', '')}_scatter.png", dpi=150); plt.close()
best = max(results, key=lambda k: (results[k][0], results[k][1]))
lines.append(f"RECOMMENDED by the pre-stated rule: {best} "
             f"(5-NN CV accuracy={results[best][0]:.4f}, trustworthiness={results[best][1]:.4f})")
(OUT / "results").mkdir(exist_ok=True)
(OUT / "results" / "results.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
