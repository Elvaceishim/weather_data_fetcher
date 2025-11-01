#!/usr/bin/env python3
"""
Generates a SHAP dependence plot showing how HUMIDITY and 
TEMPERATURE
jointly influence rain predictions. Outputs:
  - results/shap_interaction.png
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import importlib.util
import os

# Keep matplotlib caches inside repo to avoid home directory issues
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

# Load model + meta
model = joblib.load("models/rain_xgb_tuned.joblib")
booster = model.get_booster()
config = json.loads(booster.save_config())
base_score = config.get("learner", {}).get("learner_model_param", {}).get("base_score")
if base_score:
    cleaned = base_score.strip("[]")
    try:
        float(cleaned)
    except ValueError:
        cleaned = "0.5"
    config["learner"]["learner_model_param"]["base_score"] = cleaned
    booster.load_config(json.dumps(config))

meta = json.loads(Path("models/rain_xgb_tuned_meta.json").read_text())
features = meta["features"]

# Load data and rebuild features exactly like training
df = pd.read_csv("results/hourly.csv", parse_dates=["time"])

spec = importlib.util.spec_from_file_location(
    "train_xgb_tuned_final", "scripts/train_xgb_tuned_final.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
build_features = module.build_features
Xdf = build_features(df)              # same order as training
X = Xdf.values
X_sample = X[-120:] if len(X) > 120 else X
X_sample_df = pd.DataFrame(X_sample, columns=features)
X_sample_df = pd.DataFrame(X_sample, columns=features)

# Prefer TreeExplainer for XGBoost; fallback to generic Explainer if needed
try:
    explainer = shap.TreeExplainer(booster, data=X_sample)
    shap_result = explainer(X_sample)
except Exception:
    explainer = shap.Explainer(model.predict_proba, X_sample, algorithm="permutation")
    shap_result = explainer(X_sample)

# Normalize SHAP output to a 2D array aligned with feature columns
if hasattr(shap_result, "values"):
    values = shap_result.values
    if values.ndim == 3:  # multi-class, take positive class (index 1)
        values = values[:, :, 1]
    shap_values = values
else:
    shap_values = np.array(shap_result)

# Ensure sample frame matches SHAP output rows
X_plot = X_sample_df.iloc[-shap_values.shape[0]:]

Path("results").mkdir(exist_ok=True)

# 1) Dependence plot: humidity colored by temp_c (classic interaction view)
plt.figure()
shap.dependence_plot(
    "humidity",
    shap_values,
    X_plot,
    interaction_index="temp_c",
    show=False
)
plt.tight_layout()
plt.savefig("results/shap_interaction.png", dpi=300)
plt.close()

# 2) (Optional) Reverse view: temp_c colored by humidity
plt.figure()
shap.dependence_plot(
    "temp_c",
    shap_values,
    X_plot,
    interaction_index="humidity",
    show=False
)
plt.tight_layout()
plt.savefig("results/shap_interaction_rev.png", dpi=300)
plt.close()

print("✅ Saved results/shap_interaction.png and results/shap_interaction_rev.png")
