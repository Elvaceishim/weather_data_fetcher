import json
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split

RESULTS_DIR = "results"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(RESULTS_DIR, ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(RESULTS_DIR, ".cache"))

from pathlib import Path

Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

meta = json.load(open("models/rain_model_meta.json"))
clf = joblib.load("models/rain_classifier_hourly.joblib")
df = pd.read_csv("results/hourly.csv", parse_dates=["time"])

H = meta["horizon_hours"]
features = meta["features"]

precip_next = np.zeros(len(df), dtype=int)
prec = df["precip_mm"].values
for i in range(len(prec) - H):
    precip_next[i] = 1 if np.any(prec[i + 1 : i + 1 + H] > 0) else 0

df = df.iloc[: len(precip_next)].copy()
df["rain_next6h"] = precip_next[: len(df)]

X = df[features].values
y = df["rain_next6h"].values

_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
proba = clf.predict_proba(X_test)[:, 1]

precision, recall, _ = precision_recall_curve(y_test, proba)
fpr, tpr, _ = roc_curve(y_test, proba)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall")
plt.tight_layout()
plt.savefig("results/pr_curve.png")
plt.close()

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title(f"ROC (AUC={auc(fpr, tpr):.2f})")
plt.tight_layout()
plt.savefig("results/roc_curve.png")
plt.close()

print("✅ Wrote results/pr_curve.png and results/roc_curve.png")
