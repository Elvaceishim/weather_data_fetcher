import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_fscore_support
)

df = pd.read_csv("results/hourly.csv", parse_dates=["time"])

H = 6  # look-ahead horizon (hours)
precip_next = np.zeros(len(df), dtype=int)

# Use precip_mm (more inclusive than rain_mm)
prec = df["precip_mm"].values

# For each time t, look at t+1 ... t+H
for i in range(len(prec) - H):
    precip_next[i] = 1 if np.any(prec[i+1:i+1+H] > 0) else 0

# Cut tail (no full future window)
df = df.iloc[:len(precip_next) - (0)].copy()
df["rain_next6h"] = precip_next[:len(df)]


features = [
    "temp_c","humidity","cloudcover","pressure","wind_speed",
    "precip_mm","rain_mm"
]
X = df[features].values
y = df["rain_next6h"].values

# Quick sanity check
print("Class balance (0=no-rain, 1=rain-in-next6h):", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=500, class_weight="balanced"))
])
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]
pred_050 = (proba >= 0.50).astype(int)

cm = confusion_matrix(y_test, pred_050)
print("\n📊 Confusion Matrix (thr=0.50)")
print(cm)

prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, pred_050, average="binary", zero_division=0
)
try:
    auc = roc_auc_score(y_test, proba)
except ValueError:
    auc = float("nan")

print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}  ROC-AUC: {auc:.3f}")

print("\nDetailed report:")
print(classification_report(y_test, pred_050, digits=3, zero_division=0))

# Baselines
always_no = np.zeros_like(y_test)
p0, r0, f10, _ = precision_recall_fscore_support(
    y_test, always_no, average="binary", zero_division=0
)
print("\n🧠 Baseline — always 'no rain'")
print(f"Precision: {p0:.3f}  Recall: {r0:.3f}  F1: {f10:.3f}")

# Persistence baseline: if it rained in the last H hours, predict rain in next H hours
recent_rain = (
    pd.Series(df["precip_mm"])
    .rolling(window=H, min_periods=1)
    .sum()
    .shift(1)
    .fillna(0)
    > 0
).astype(int).values
prev6_test = recent_rain[-len(y_test):]
pp, rp, f1p, _ = precision_recall_fscore_support(y_test, prev6_test, average="binary", zero_division=0)
print("\n🧠 Baseline — persistence (prev 6h)")
print(f"Precision: {pp:.3f}  Recall: {rp:.3f}  F1: {f1p:.3f}")

# Threshold tuning
thr_recall = 0.35
thr_precision = 0.65

pred_recall = (proba >= thr_recall).astype(int)
pred_precision = (proba >= thr_precision).astype(int)

pr_recall, rc_recall, f1_recall, _ = precision_recall_fscore_support(
    y_test, pred_recall, average="binary", zero_division=0
)
pr_precision, rc_precision, f1_precision, _ = precision_recall_fscore_support(
    y_test, pred_precision, average="binary", zero_division=0
)

print(f"\n🎛️ Threshold {thr_recall:.2f} → Precision: {pr_recall:.3f}  Recall: {rc_recall:.3f}  F1: {f1_recall:.3f}")
print(f"🎛️ Threshold {thr_precision:.2f} → Precision: {pr_precision:.3f}  Recall: {rc_precision:.3f}  F1: {f1_precision:.3f}")

# Save model
import joblib
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/rain_classifier_hourly.joblib")
print("\n💾 Saved: models/rain_classifier_hourly.joblib")

meta = {
    "horizon_hours": H,
    "features": features,
    "thresholds": {
        "default": 0.50,
        "high_recall": thr_recall,
        "high_precision": thr_precision,
    },
    "metrics": {
        "default": {"precision": float(prec), "recall": float(rec), "f1": float(f1)},
        "high_recall": {
            "precision": float(pr_recall),
            "recall": float(rc_recall),
            "f1": float(f1_recall),
        },
        "high_precision": {
            "precision": float(pr_precision),
            "recall": float(rc_precision),
            "f1": float(f1_precision),
        },
        "baseline_persistence": {
            "precision": float(pp),
            "recall": float(rp),
            "f1": float(f1p),
        },
    },
}

with open("models/rain_model_meta.json", "w") as fh:
    import json
    json.dump(meta, fh, indent=2)

print("📝 Saved: models/rain_model_meta.json")
