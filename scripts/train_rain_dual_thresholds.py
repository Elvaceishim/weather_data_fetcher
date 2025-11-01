
import os, json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve, confusion_matrix, precision_recall_fscore_support,
    classification_report, roc_auc_score
)

# 1) Load hourly data
df = pd.read_csv("results/hourly.csv", parse_dates=["time"])

# 2) Label (H = 12)
H = 12
prec = df["precip_mm"].values
rain_next = np.zeros(len(df), dtype=int)
for i in range(len(prec) - H):
    rain_next[i] = 1 if np.any(prec[i+1:i+1+H] > 0) else 0

df = df.iloc[:len(prec)].copy()
df["rain_next12h"] = rain_next[:len(df)]
df = df.iloc[:-H].copy()

# 3) Feature engineering
base_cols = ["temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"]
for col in base_cols:
    df[f"d_{col}"] = df[col].diff()                    # 1h change
    df[f"ma3_{col}"] = df[col].rolling(3).mean()       # 3h rolling mean

df = df.dropna().reset_index(drop=True)

base = base_cols
dels = [f"d_{c}" for c in base_cols]
mas  = [f"ma3_{c}" for c in base_cols]

features = base + dels + mas
X = df[features].values
y = df["rain_next12h"].values

# 4) Time-aware split (no shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, shuffle=False
)

# 5) Pipeline (scale + logistic regression)
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
clf.fit(X_train, y_train)

# 6) Evaluate + Precision–Recall curve
proba = clf.predict_proba(X_test)[:, 1]
pred_050 = (proba >= 0.50).astype(int)

cm_050 = confusion_matrix(y_test, pred_050)
P_050, R_050, F1_050, _ = precision_recall_fscore_support(y_test, pred_050, average="binary", zero_division=0)
try:
    auc = roc_auc_score(y_test, proba)
except ValueError:
    auc = float("nan")

print("Class balance (test y):", np.bincount(y_test))
print("\nDefault (0.50):", {
    "threshold": 0.50, "precision": float(P_050), "recall": float(R_050),
    "f1": float(F1_050), "auc": float(auc), "cm": cm_050.tolist()
})

prec, rec, thr = precision_recall_curve(y_test, proba)


def compute_metrics(proba_vals, true_vals):
    thresholds = np.unique(np.concatenate(([0.0], proba_vals)))
    results = []
    for th in thresholds:
        preds = (proba_vals >= th).astype(int)
        P, R, F1, _ = precision_recall_fscore_support(true_vals, preds, average="binary", zero_division=0)
        cm = confusion_matrix(true_vals, preds)
        results.append(
            dict(
                threshold=float(th),
                precision=float(P),
                recall=float(R),
                f1=float(F1),
                auc=float(auc),
                cm=cm.tolist(),
            )
        )
    results.sort(key=lambda x: x["threshold"])
    return results


def pick_with_constraints(metrics, min_precision, min_recall):
    candidates = [m for m in metrics if m["precision"] >= min_precision and m["recall"] >= min_recall]
    if not candidates:
        candidates = metrics
    return max(candidates, key=lambda m: (m["f1"], m["threshold"]))


all_metrics = compute_metrics(proba, y_test)
res_default = max(all_metrics, key=lambda m: (m["f1"], m["threshold"]))
res_recall = pick_with_constraints(all_metrics, min_precision=0.65, min_recall=0.85)
res_prec = pick_with_constraints(all_metrics, min_precision=0.90, min_recall=0.45)

print("\nChosen thresholds:",
      "\n default:", res_default["threshold"],
      "\n high_recall:", res_recall["threshold"],
      "\n high_precision:", res_prec["threshold"])

print("\nHigh-Recall:", res_recall)
print("\nHigh-Precision:", res_prec)

# 7) Save model + metadata
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/rain_classifier_hourly.joblib")

meta = {
    "features": features,
    "horizon_hours": H,
    "thresholds": {
        "default": res_default["threshold"],
        "high_recall": res_recall["threshold"],
        "high_precision": res_prec["threshold"]
    },
    "metrics": {
        "default": res_default,
        "high_recall": res_recall,
        "high_precision": res_prec
    }
}
with open("models/rain_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n💾 Saved models/rain_classifier_hourly.joblib and models/rain_model_meta.json")
