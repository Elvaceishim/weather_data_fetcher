import pandas as pd

df = pd.read_csv("results/summary.csv")
print("\n=== HEAD ===")
print(df.head())
print("\n=== DESCRIBE ===")
print(df.describe())
print("\n=== COLUMNS ===")
print(df.columns)
print("\n=== MISSING VALUES ===")
print(df.isna().sum())
