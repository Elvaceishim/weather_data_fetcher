import os, json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_recall_curve, confusion_matrix, precision_recall_fscore_support,
    classification_report, roc_auc_score
)

# ------- Load hourly data -------
df = pd.read_csv("results/hourly.csv", parse_dates=["time"])

# ------- Label: any rain in next H hours? -------
H = 6
prec = df["precip_mm"].values
rain_next = np.zeros(len(df), dtype=int)
for i in range(len(prec) - H):
    rain_next[i] = 1 if np.any(prec[i+1:i+1+H] > 0) else 0
df = df.iloc[:len(prec) - 0].copy()
df["rain_next6h"] = rain_next[:len(df)]

df = df.iloc[:-H].copy()

# ------- Feature engineering -------
for col in ["temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"]:
    df[f"d_{col}"] = df[col].diff()                     # 1h change
    df[f"ma3_{col}"] = df[col].rolling(3).mean()        # 3h mean

df = df.dropna().reset_index(drop=True)

base = ["temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"]
dels = [f"d_{c}" for c in base]
mas  = [f"ma3_{c}" for c in base]

features = base + dels + mas
X = df[features].values
y = df["rain_next6h"].values

# ------- Time-aware split -------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

# ------- Pipeline -------
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
clf.fit(X_train, y_train)

# ------- Scores & curve -------
proba = clf.predict_proba(X_test)[:, 1]

# Precision-Recall curve to choose thresholds for both modes
prec, rec, thr = precision_recall_curve(y_test, proba)

def pick_threshold_for_recall(target):
    # highest threshold that achieves at least target recall
    idx = np.where(rec >= target)[0]
    if len(idx) == 0: return 0.5  # fallback
    j = idx[-1]  # highest precision point for that recall segment
    return thr[j-1] if j > 0 and j-1 < len(thr) else thr[0]

def pick_threshold_for_precision(target):
    idx = np.where(prec >= target)[0]
    # thresholds array is len-1 compared to prec/rec
    if len(idx) == 0: return 0.5
    j = idx[0]  # first point reaching that precision
    return thr[j-1] if j > 0 and j-1 < len(thr) else thr[0]

thr_recall = pick_threshold_for_recall(0.80)     # High-Recall target (80%)
thr_precision = pick_threshold_for_precision(0.90)  # High-Precision target (90%)

def eval_at(th):
    pred = (proba >= th).astype(int)
    cm = confusion_matrix(y_test, pred)
    P, R, F1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    try: auc = roc_auc_score(y_test, proba)
    except: auc = float("nan")
    return dict(threshold=float(th), precision=float(P), recall=float(R), f1=float(F1), auc=float(auc), cm=cm.tolist())

res_default = eval_at(0.50)
res_recall  = eval_at(thr_recall)
res_prec    = eval_at(thr_precision)

print("Class balance (test y):", np.bincount(y_test))
print("\nDefault (0.50):", res_default)
print("\nHigh-Recall:", res_recall)
print("\nHigh-Precision:", res_prec)

# Save model + thresholds
os.makedirs("models", exist_ok=True)
import joblib
joblib.dump(clf, "models/rain_classifier_hourly.joblib")

meta = {
    "features": features,
    "horizon_hours": H,
    "thresholds": {
        "default": 0.50,
        "high_recall": res_recall["threshold"],
        "high_precision": res_prec["threshold"]
    },
    "metrics": {
        "default": res_default, "high_recall": res_recall, "high_precision": res_prec
    }
}
with open("models/rain_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("\n💾 Saved models/rain_classifier_hourly.joblib and models/rain_model_meta.json")
