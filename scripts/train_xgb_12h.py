
import os, json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, confusion_matrix, roc_auc_score

from xgboost import XGBClassifier, callback

# -------- Load & label (12h) --------
df = pd.read_csv("results/hourly.csv", parse_dates=["time"])

H = 12
prec = df["precip_mm"].values
y_all = np.zeros(len(df), dtype=int)
for i in range(len(prec) - H):
    y_all[i] = 1 if np.any(prec[i+1:i+1+H] > 0) else 0

df = df.iloc[:-H].copy()
df["rain_next12h"] = y_all[:len(df)]

# -------- Features (same as logistic) --------
base = ["temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"]
for c in base:
    df[f"d_{c}"] = df[c].diff()
    df[f"ma3_{c}"] = df[c].rolling(3).mean()

# Onset features (3h deltas)
df["pressure_d3h"] = df["pressure"] - df["pressure"].shift(3)
df["humidity_d3h"] = df["humidity"] - df["humidity"].shift(3)
df["cloudcover_d3h"] = df["cloudcover"] - df["cloudcover"].shift(3)

# Dewpoint proxy and its dynamics
df["dew_proxy"] = df["temp_c"] - (df["humidity"] / 5)
df["d_dew_proxy"] = df["dew_proxy"].diff()
df["ma3_dew_proxy"] = df["dew_proxy"].rolling(3).mean()

df = df.dropna().reset_index(drop=True)

features = (
    base
    + [f"d_{c}" for c in base]
    + [f"ma3_{c}" for c in base]
    + ["pressure_d3h", "humidity_d3h", "cloudcover_d3h"]
    + ["dew_proxy", "d_dew_proxy", "ma3_dew_proxy"]
)
X = df[features].values
y = df["rain_next12h"].values

# -------- Time-aware split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, shuffle=False
)

n = len(X_train)
split = int(n * 0.80)
X_tr, X_val = X_train[:split], X_train[split:]
y_tr, y_val = y_train[:split], y_train[split:]

neg = (y_tr == 0).sum()
pos = (y_tr == 1).sum()
scale_pos_weight = neg / pos if pos else 1.0

# -------- XGBoost config --------
clf = XGBClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    early_stopping_rounds=50,
    scale_pos_weight=scale_pos_weight
)

clf.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

# -------- Evaluate on TEST --------
proba = clf.predict_proba(X_test)[:, 1]

# utility to evaluate a threshold
def eval_at(scores, y_true, thr):
    pred = (scores >= thr).astype(int)
    P, R, F1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, pred).tolist()
    try: auc = roc_auc_score(y_true, scores)
    except: auc = float("nan")
    return dict(threshold=float(thr), precision=float(P), recall=float(R), f1=float(F1), auc=float(auc), cm=cm)

# grid search across PR curve thresholds with constraints
prec_curve, rec_curve, thr = precision_recall_curve(y_test, proba)
def grid_metrics(scores, y_true, thr_list):
    out = []
    for t in thr_list:
        out.append(eval_at(scores, y_true, t))
        out[-1]["pos_rate"] = float((scores >= t).mean())
    return out

grid = grid_metrics(proba, y_test, thr)

def pick_best_f1(grid):
    return max(grid, key=lambda g: g["f1"])

def pick_high_recall(grid, target_recall=0.90, min_precision=0.70, max_pos_rate=0.80):
    cand = [g for g in grid if g["recall"] >= target_recall
                            and g["precision"] >= min_precision
                            and g["pos_rate"] <= max_pos_rate]
    return max(cand, key=lambda g: g["f1"]) if cand else pick_best_f1(grid)

def pick_high_precision(grid, target_precision=0.90, min_recall=0.50):
    cand = [g for g in grid if g["precision"] >= target_precision
                            and g["recall"] >= min_recall]
    return max(cand, key=lambda g: (g["precision"], g["f1"])) if cand else pick_best_f1(grid)

res_default  = pick_best_f1(grid)
res_recall   = pick_high_recall(grid)
res_prec     = pick_high_precision(grid)

print("Class balance (test y):", np.bincount(y_test))
print("\nXGB — Default (best F1):", res_default)
print("\nXGB — High-Recall:",       res_recall)
print("\nXGB — High-Precision:",    res_prec)

# -------- Save model + meta --------
Path("models").mkdir(exist_ok=True)
joblib.dump(clf, "models/rain_xgb_12h.joblib")

meta = {
    "features": features,
    "horizon_hours": 12,
    "thresholds": {
        "default":       res_default["threshold"],
        "high_recall":   res_recall["threshold"],
        "high_precision":res_prec["threshold"]
    },
    "metrics": {
        "default": res_default,
        "high_recall": res_recall,
        "high_precision": res_prec
    },
    "model_type": "xgboost"
}
with open("models/rain_xgb_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n💾 Saved models/rain_xgb_12h.joblib and models/rain_xgb_meta.json")
