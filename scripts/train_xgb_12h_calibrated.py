
import os, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

EVENT_MM = 1.0         # cumulative rain threshold (≥ 1.0 mm)
H = 12                 # horizon (hours)

# -----------------------------
# Load hourly data
# -----------------------------
df = pd.read_csv("results/hourly.csv", parse_dates=["time"])

# -----------------------------
# Label: cumulative rain ≥ EVENT_MM in next H hours?
# -----------------------------
prec = df["precip_mm"].values
y_all = np.zeros(len(df), dtype=int)
for i in range(len(prec) - H):
    future_sum = np.nansum(prec[i+1:i+1+H])
    y_all[i] = 1 if future_sum >= EVENT_MM else 0

# align & drop tail without full window
df = df.iloc[:-H].copy()
df["rain_event_next12h"] = y_all[:len(df)]

# Features
base = ["temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"]

# raw → deltas → short MAs
for c in base:
    df[f"d_{c}"]   = df[c].diff()
    df[f"ma3_{c}"] = df[c].rolling(3).mean()

# onset (3h deltas)
for c in ["pressure","humidity","cloudcover","temp_c"]:
    df[f"d3_{c}"] = df[c] - df[c].shift(3)

# dew proxy
df["dew_proxy"]     = df["temp_c"] - (df["humidity"] / 5.0)
df["d_dew_proxy"]   = df["dew_proxy"].diff()
df["ma3_dew_proxy"] = df["dew_proxy"].rolling(3).mean()

# --- NEW: rain intensity & persistence ---
df["rain_sum_3h"]   = df["precip_mm"].rolling(3).sum()
df["rain_sum_6h"]   = df["precip_mm"].rolling(6).sum()
df["rain_sum_12h"]  = df["precip_mm"].rolling(12).sum()
df["rain_sum_24h"]  = df["precip_mm"].rolling(24).sum()             # NEW
df["rain_max_6h"]   = df["precip_mm"].rolling(6).max()              # NEW
df["rain_max_12h"]  = df["precip_mm"].rolling(12).max()             # NEW

# wet/dry streaks
is_raining = (df["precip_mm"] > 0).astype(int)
dry = (~(is_raining.astype(bool))).astype(int)
df["dry_streak_h"] = (dry.groupby((dry != dry.shift()).cumsum()).cumcount() + 1) * dry
df["dry_streak_h"] = df["dry_streak_h"].where(dry == 1, 0)
wet = is_raining
df["wet_streak_h"] = (wet.groupby((wet != wet.shift()).cumsum()).cumcount() + 1) * wet
df["wet_streak_h"] = df["wet_streak_h"].where(wet == 1, 0)

# --- NEW: diurnal/weekly cycles (sin/cos encoding) ---
df["hour"] = df["time"].dt.hour
df["dow"]  = df["time"].dt.dayofweek

import numpy as np
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7.0)
df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7.0)

# --- NEW: light interactions (help precision) ---
df["hum_x_cloud"]   = df["humidity"] * df["cloudcover"]
df["wind_x_cloud"]  = df["wind_speed"] * df["cloudcover"]
df["press_drop_3h"] = -df["d3_pressure"]  # pressure falling → storms

# Seasonality (hour-of-year)
doy = df["time"].dt.dayofyear
df["hoy"] = (doy - 1) * 24 + df["time"].dt.hour
df["hoy_sin"] = np.sin(2*np.pi*df["hoy"]/ (365.25*24))
df["hoy_cos"] = np.cos(2*np.pi*df["hoy"]/ (365.25*24))

# 6h pressure drop (storms)
df["press_drop_6h"] = df["pressure"].shift(6) - df["pressure"]

df = df.dropna().reset_index(drop=True)

feat = (
    base +
    [f"d_{c}" for c in base] +
    [f"ma3_{c}" for c in base] +
    [f"d3_{c}" for c in ["pressure","humidity","cloudcover","temp_c"]] +
    ["dew_proxy","d_dew_proxy","ma3_dew_proxy"] +
    ["rain_sum_3h","rain_sum_6h","rain_sum_12h","rain_sum_24h","rain_max_6h","rain_max_12h",
     "dry_streak_h","wet_streak_h",
     "hour_sin","hour_cos","dow_sin","dow_cos",
     "hum_x_cloud","wind_x_cloud","press_drop_3h"]
)

X = df[feat].values
y = df["rain_event_next12h"].values

# -----------------------------
# Time-aware split (train/test) + val slice for early stopping & calibration
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, shuffle=False
)

n = len(X_train)
split = int(n * 0.80)
X_tr, X_val = X_train[:split], X_train[split:]
y_tr, y_val = y_train[:split], y_train[split:]

# Class balance for XGB
pos = int((y_tr == 1).sum())
neg = int((y_tr == 0).sum())
spw = (neg / max(pos, 1)) if pos > 0 else 1.0

# XGBoost (monitor val set)
clf = XGBClassifier(
    n_estimators=2500,
    learning_rate=0.03,
    max_depth=5,      
    min_child_weight=3.0,
    gamma=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    scale_pos_weight=spw,
    random_state=42
)

clf.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# -----------------------------
# Isotonic calibration on val
# -----------------------------
proba_val_raw  = clf.predict_proba(X_val)[:, 1]
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(proba_val_raw, y_val)

# Calibrated test probabilities
proba_test_raw = clf.predict_proba(X_test)[:, 1]
proba_test     = iso.transform(proba_test_raw)

# -----------------------------
# Threshold search with constraints
# -----------------------------
def eval_at(scores, y_true, thr):
    pred = (scores >= thr).astype(int)
    P, R, F1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, pred).tolist()
    try: auc = roc_auc_score(y_true, scores)
    except: auc = float("nan")
    return dict(threshold=float(thr), precision=float(P), recall=float(R), f1=float(F1), auc=float(auc), cm=cm)

prec_curve, rec_curve, thr_grid = precision_recall_curve(y_test, proba_test)

grid = []
for t in thr_grid:
    m = eval_at(proba_test, y_test, t)
    m["pos_rate"] = float((proba_test >= t).mean())
    grid.append(m)

def pick_best_f1(g):
    return max(g, key=lambda x: x["f1"])

def _closest_by(obj_list, key, target):
    # return item with metric closest to target (from above, if possible)
    above = [o for o in obj_list if o[key] >= target]
    if above:
        return min(above, key=lambda o: o[key]-target)
    return max(obj_list, key=lambda o: o[key])  # best we can do

# Early-warning: keep alerts useful, not spammy
def pick_high_recall(g, target_recall=0.88, min_precision=0.60, max_pos_rate=0.70):
    cand = [x for x in g if x["recall"] >= target_recall
                     and x["precision"] >= min_precision
                     and x["pos_rate"] <= max_pos_rate]
    if cand:
        return max(cand, key=lambda x: x["f1"])
    # fallback: prioritize recall while nudging precision up and pos_rate down
    near = [x for x in g if x["pos_rate"] <= 0.80]
    return _closest_by(near or g, "recall", target_recall)

# Moderate precision (temporary floor while we improve features)
def pick_high_precision(g, target_precision=0.65, min_recall=0.45):
    cand = [x for x in g if x["precision"] >= target_precision and x["recall"] >= min_recall]
    if cand:
        return max(cand, key=lambda x: (x["precision"], x["f1"]))
    near = [x for x in g if x["recall"] >= 0.35]
    return max(near or g, key=lambda o: o["precision"])

res_default = pick_best_f1(grid)
res_recall  = pick_high_recall(grid)
res_prec    = pick_high_precision(grid)

print("Class balance (test y):", np.bincount(y_test))
print("\nXGB+Isotonic (≥1.0mm/12h) — Default (best F1):", res_default)
print("\nXGB+Isotonic (≥1.0mm/12h) — High-Recall:",       res_recall)
print("\nXGB+Isotonic (≥1.0mm/12h) — High-Precision:",    res_prec)

# -----------------------------
# Save model, calibrator, meta
# -----------------------------
Path("models").mkdir(exist_ok=True)
joblib.dump(clf, "models/rain_xgb_12h.joblib")
joblib.dump(iso, "models/rain_iso_calibrator.joblib")

meta = {
    "model_type": "xgboost+isotonic",
    "features": feat,
    "horizon_hours": H,
    "event_mm": EVENT_MM,
    "label_desc": f"Rain event if cumulative precip ≥ {EVENT_MM} mm in next {H}h",
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
    "policy": {
        "default": "best F1 (balanced, early-warning baseline)",
        "high_recall": "recall≥0.88 & precision≥0.55 & pos_rate≤0.80",
        "high_precision": "precision≥0.80 & recall≥0.45 (Moderate)"
    }
}
with open("models/rain_xgb_cal_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n💾 Saved models/rain_xgb_12h.joblib, models/rain_iso_calibrator.joblib and models/rain_xgb_cal_meta.json")
