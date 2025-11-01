import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

meta = json.loads(Path("models/rain_xgb_tuned_meta.json").read_text())
thr = meta["thresholds"]
H = meta["horizon_hours"]

df = pd.read_csv("results/hourly.csv", parse_dates=["time"])

# make labels (>=1.0mm in next H hours)
prec = df["precip_mm"].values
y = np.zeros(len(df), dtype=int)
for i in range(len(prec) - H):
    y[i] = 1 if np.nansum(prec[i+1:i+1+H]) >= meta["event_mm"] else 0
y = y[:-H]
dfX = df.iloc[:-H].copy()

# rebuild features exactly like training
# local import
import importlib.util
import types

def load_build_features():
    spec = importlib.util.spec_from_file_location("train_xgb_tuned_final", Path("scripts/train_xgb_tuned_final.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module.build_features

build_features = load_build_features()
Xdf = build_features(dfX)
X = Xdf.values
y = y[-len(X):]  # align

import joblib
clf = joblib.load("models/rain_xgb_tuned.joblib")
p = clf.predict_proba(X)[:,1]

def report(name, t):
    pred = (p >= t).astype(int)
    P, R, F1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y, pred).tolist()
    rate = float(pred.mean())
    print(f"{name:<10} thr={t:.3f} | P={P:.3f} R={R:.3f} F1={F1:.3f} | alerts={rate:.2%} | cm={cm}")

report("default",   thr["default"])
report("recall",    thr["high_recall"])
report("precision", thr["high_precision"])
