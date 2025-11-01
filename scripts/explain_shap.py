#!/usr/bin/env python3
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np
import os

# ensure matplotlib cache lives inside repo
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

# === Load model + metadata ===
model = joblib.load("models/rain_xgb_tuned.joblib")
meta = json.load(open("models/rain_xgb_tuned_meta.json"))
features = meta["features"]

# === Load data ===
df = pd.read_csv("results/hourly.csv", parse_dates=["time"])

# Rebuild features exactly like training
import importlib.util

spec = importlib.util.spec_from_file_location(
    "train_xgb_tuned_final", Path("scripts/train_xgb_tuned_final.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
build_features = module.build_features

Xdf = build_features(df)
X = Xdf.values.astype(np.float32)

# Use last 500 samples for analysis (avoid overkill)
X_sample = X[-200:]

# === SHAP Explainer ===
explainer = shap.Explainer(model.predict_proba, X_sample, algorithm="permutation")
shap_values = explainer(X_sample)

# === Global importance ===
Path("results").mkdir(exist_ok=True)
plt.figure()
shap.summary_plot(shap_values, X_sample, 
feature_names=features, show=False)
plt.tight_layout()
plt.savefig("results/shap_summary.png", dpi=300)
plt.close()

# === Bar chart version ===
plt.figure()
shap.summary_plot(shap_values, X_sample, 
feature_names=features, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("results/shap_top.png", dpi=300)
plt.close()

print("✅ SHAP visualisations saved: results/shap_summary.png and results/shap_top.png")
