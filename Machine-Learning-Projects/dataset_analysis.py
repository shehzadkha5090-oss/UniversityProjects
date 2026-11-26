"""Dataset analysis for ML Lab Projects v4 (Student Performance).

Loads the single shared dataset (UCI student-por.csv, semicolon-delimited),
prints its shape, the derived target distribution, missingness, and the
key subgroup pass rates used across the 15 projects. Run with the project
venv, e.g.:  .venv/bin/python dataset_analysis.py
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "shared_dataset" / "dataset.csv"

df = pd.read_csv(DATA, sep=";")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"Missing values (total): {int(df.isna().sum().sum())}")
print(f"Columns with any missing value: {df.columns[df.isna().any()].tolist()}")

df["pass"] = (df["G3"] >= 10).astype(int)
print(f"\nTarget pass = (G3 >= 10): pass {df['pass'].sum()} "
      f"({df['pass'].mean() * 100:.1f}%), fail {(1 - df['pass']).sum()} "
      f"({(1 - df['pass']).mean() * 100:.1f}%)")
print(f"G3: mean {df['G3'].mean():.2f}, std {df['G3'].std():.2f}, "
      f"min {df['G3'].min()}, max {df['G3'].max()}")
print("\nPass rate by school:")
print(df.groupby("school")["pass"].agg(["count", "mean"]).round(3))
print("\nPass rate by sex:")
print(df.groupby("sex")["pass"].agg(["count", "mean"]).round(3))
print("\nGrade correlations with G3:")
print(df[["G1", "G2", "G3"]].corr().round(3))
print("\nNumeric means:")
print(df[["age", "absences", "G1", "G2", "G3"]].mean().round(2))
