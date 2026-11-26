# ML Lab Projects — all 20 labs

20 Machine Learning lab projects (Labs 1–20 from the course PDF), all sharing one
dataset: **UCI Student Performance** (`student-por.csv`, 649 Portuguese-course
students). Target: `pass = (G3 >= 10)` — 84.6% pass, 15.4% fail.

Full write-up, per-project results, and honest caveats (including the labelled
adaptations for Labs 7, 9, 11, 13) are in [`master_report.md`](master_report.md).

## Layout

- `shared_dataset/` — the dataset (`dataset.csv`), its documentation, and UCI's `student.txt`
- `project_01/` … `project_20/` — one folder per lab, each with `main.py`, `README.md`,
  `results/results.txt`, and `visualizations/*.png`
- `dataset_analysis.py` — reproduces the dataset facts
- `requirements.txt` — shared dependencies

## Run

```bash
pip install -r requirements.txt
python dataset_analysis.py
cd project_01 && python main.py
```

Each `main.py` loads only `../shared_dataset/dataset.csv`, so projects run
independently in any order. Re-running a project regenerates its
`results/results.txt` and charts.

## Note on generated files

The charts in `visualizations/` and `project_20/results/model.joblib` are
generated outputs — running each project's `main.py` recreates them.
