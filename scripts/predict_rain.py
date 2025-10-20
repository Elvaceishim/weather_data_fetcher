import sys, json, joblib
import numpy as np
import pandas as pd

# Load latest hour from results/hourly.csv, predict next 6h rain
df = pd.read_csv("results/hourly.csv", parse_dates=["time"])
row = df.iloc[-1:].copy()

# Rebuild features like in training
for col in ["temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"]:
    row[f"d_{col}"] = df[col].diff().iloc[-1]
    row[f"ma3_{col}"] = df[col].rolling(3).mean().iloc[-1]

features = ["temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"]
features += [f"d_{c}" for c in features]
features += [f"ma3_{c}" for c in ["temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"]]

# Load model + meta
clf = joblib.load("models/rain_classifier_hourly.joblib")
meta = json.load(open("models/rain_model_meta.json"))
X = row[meta["features"]].values

proba = float(clf.predict_proba(X)[0,1])
thr_r = meta["thresholds"]["high_recall"]
thr_p = meta["thresholds"]["high_precision"]

print(f"Latest hour: {row['time'].iloc[0]}")
print(f"P(rain next {meta['horizon_hours']}h) = {proba:.3f}")
print(f"High-Recall mode:   {'RAIN' if proba>=thr_r else 'No rain'} (thr={thr_r:.2f})")
print(f"High-Precision mode:{'RAIN' if proba>=thr_p else 'No rain'} (thr={thr_p:.2f})")
