import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_fscore_support
)

# 1) Load data
df = pd.read_csv("results/daily.csv")

# 2) Label: "will it rain tomorrow?"
df["precip_tomorrow"] = df["precip_mm"].shift(-1)
df = df.dropna()  # drop last row without tomorrow

df["rain_tomorrow"] = (df["precip_tomorrow"] > 0).astype(int)

# 3) Features (start small & meaningful)
# Today's conditions → tomorrow rain?
features = [
    "temp_max_c",
    "temp_min_c",
    "cloudcover",
    "wind_speed",
    "humidity_max",
    "humidity_min",
    "precip_mm",        # rain today often implies rain persists
]
X = df[features].values
y = df["rain_tomorrow"].values

# 4) Time-aware split (no shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

# 5) Pipeline: scale → logistic regression
# class_weight="balanced" helps when rain/no-rain is imbalanced
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=200, 
class_weight="balanced"))
])

clf.fit(X_train, y_train)

# 6) Predictions
proba = clf.predict_proba(X_test)[:, 1]       # P(rain)
pred_default = (proba >= 0.5).astype(int)     # default threshold

# 7) Metrics
labels = [0, 1]
cm = confusion_matrix(y_test, pred_default, labels=labels)
tn, fp, fn, tp = cm.ravel()

prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, pred_default, average="binary", zero_division=0
)
auc = (
    roc_auc_score(y_test, proba)
    if len(np.unique(y_test)) > 1 and len(np.unique(proba)) > 1
    else float("nan")
)

print("📊 Confusion Matrix (threshold=0.50)")
print(cm)
auc_str = f"{auc:.3f}" if np.isfinite(auc) else "n/a"
print(f"\nPrecision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}  ROC-AUC: {auc_str}")

print("\nDetailed report:")
print(classification_report(y_test, pred_default, digits=3, zero_division=0, labels=labels))

# 8) Baselines for sanity
# (a) Always predict 'no rain'
always_no = np.zeros_like(y_test)
prec0, rec0, f10, _ = precision_recall_fscore_support(
    y_test, always_no, average="binary", zero_division=0
)
print("\n⚠️ Baseline — always 'no rain'")
print(f"Precision: {prec0:.3f}  Recall: {rec0:.3f}  F1: {f10:.3f}")

# (b) Tomorrow-same-as-today (persistence on rainfall)
today_rain = (df["precip_mm"].values[-len(y_test)-1:-1] > 0).astype(int)
precp, recp, f1p, _ = precision_recall_fscore_support(
    y_test, today_rain, average="binary", zero_division=0
)
print("\n🧠 Baseline — 'tomorrow rain = today rain'")
print(f"Precision: {precp:.3f}  Recall: {recp:.3f}  F1: {f1p:.3f}")

# 9) Optional: pick a more recall-friendly threshold (catch more rain days)
thr = 0.35
pred_tuned = (proba >= thr).astype(int)
prec_t, rec_t, f1_t, _ = precision_recall_fscore_support(
    y_test, pred_tuned, average="binary", zero_division=0
)
print(f"\n🎛️  Threshold {thr:.2f} → Precision: {prec_t:.3f}  Recall: {rec_t:.3f}  F1: {f1_t:.3f}")

# 10) Save model
import joblib
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/rain_classifier.joblib")
print("\n💾 Saved: models/rain_classifier.joblib")
