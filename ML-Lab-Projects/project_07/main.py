"""Lab 8 Anomaly Detection — UCI Student Performance (student-por.csv).
UNSUPERVISED ONLY: no labels are used for training. IsolationForest and
LocalOutlierFactor are run on the FULL 649 rows (small data, no subsample).
Features: all columns except G3. Validation is PROXY-ONLY: there are no
ground-truth anomalies, so flagged vs unflagged comparisons on fail rate,
mean G3 and mean absences are descriptive, NOT accuracy."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parents[1]
OUT, VIS = BASE / "project_07" / "results", BASE / "project_07" / "visualizations"
OUT.mkdir(parents=True, exist_ok=True); VIS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "shared_dataset" / "dataset.csv", sep=";")
X = df.drop(columns=["G3"])
num = X.select_dtypes(include="number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
Xa = pre.fit_transform(X)
if hasattr(Xa, "toarray"): Xa = Xa.toarray()
CONTAM = 0.10
iso = IsolationForest(contamination=CONTAM, random_state=42, n_estimators=200)
iso_flag = iso.fit_predict(Xa) == -1
lof = LocalOutlierFactor(contamination=CONTAM, n_neighbors=20)
lof_flag = lof.fit_predict(Xa) == -1
inter = int((iso_flag & lof_flag).sum()); union = int((iso_flag | lof_flag).sum())
jaccard = inter / union if union else 0.0

y_fail = (df["G3"] < 10).astype(int)
def proxy(flag):
    return dict(n=int(flag.sum()), fail_rate=float(y_fail[flag].mean()),
                mean_G3=float(df.loc[flag, "G3"].mean()),
                mean_absences=float(df.loc[flag, "absences"].mean()))
p_iso, p_lof, p_both, p_none = proxy(iso_flag), proxy(lof_flag), proxy(iso_flag & lof_flag), proxy(~(iso_flag | lof_flag))
lines = [
 "Lab 8 Anomaly Detection — UCI Student Performance (student-por.csv)",
 f"Rows: 649 (full data, no subsample); features: all except G3; contamination={CONTAM}",
 f"IsolationForest flagged: {int(iso_flag.sum())} rows",
 f"LocalOutlierFactor flagged: {int(lof_flag.sum())} rows",
 f"Both flagged: {inter}; union: {union}; Jaccard overlap: {jaccard:.4f}",
 f"PROXY-ONLY validation (no ground-truth anomalies; NOT accuracy):",
 f"  IsolationForest flagged: {p_iso}; unflagged-by-either: {p_none}",
 f"  LOF flagged: {p_lof}",
 f"  Flagged by both: {p_both}",
 "Interpretation: flagged rows are described only by these proxy differences;",
 "  a higher fail rate / lower G3 among flagged rows does not prove they are true anomalies.",
]
(OUT / "results.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

pca = PCA(n_components=2, random_state=42).fit_transform(Xa)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(pca[~(iso_flag | lof_flag), 0], pca[~(iso_flag | lof_flag), 1], s=12, alpha=0.5, label="Unflagged")
ax.scatter(pca[iso_flag & ~lof_flag, 0], pca[iso_flag & ~lof_flag, 1], s=30, label="IsolationForest only")
ax.scatter(pca[lof_flag & ~iso_flag, 0], pca[lof_flag & ~iso_flag, 1], s=30, label="LOF only")
ax.scatter(pca[iso_flag & lof_flag, 0], pca[iso_flag & lof_flag, 1], s=40, marker="X", label="Both")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Lab 8: PCA scatter of flagged vs unflagged rows")
ax.legend(); fig.tight_layout(); fig.savefig(VIS / "anomaly_pca_scatter.png", dpi=150)
print("Saved anomaly_pca_scatter.png")
