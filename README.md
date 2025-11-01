# Weather Data Fetcher — Automated Data Pipeline

Fetch daily Lagos (or any city) weather data using **Open-Meteo API**, process it with **Python**, and automate the full workflow via **Bash + Makefile**.

---

## Project Overview

This project demonstrates a clean, reproducible workflow for data automation — the same principles used in ML and DevOps pipelines.

**Pipeline Steps**

1. Download daily weather JSON from Open-Meteo
2. Parse, validate, and summarize data in Python
3. Generate text + CSV summaries (and optional plots)
4. Automate everything via a single `make all` command

---

## Charts

## 🌧️ Rain Warning (next 6 hours)

Predict **whether it will rain in the next 6 hours** from hourly observations (temperature, humidity, pressure, wind, cloud cover, precipitation).

| Mode           | Threshold | Precision | Recall | When to use            |
| -------------- | --------- | --------- | ------ | ---------------------- |
| Default        | 0.50      | 0.71      | 0.70   | Balanced alerts        |
| High recall    | 0.35      | 0.68      | 0.84   | Better safe than sorry |
| High precision | 0.65      | 0.79      | 0.50   | Only warn if confident |

### Train once

```bash
make hourly
make rain-train
make rain-now
python scripts/train_rain_dual_thresholds.py
python scripts/plot_pr_roc.py  # refresh PR/ROC charts
```

This produces:

- `models/rain_classifier_hourly.joblib`
- `models/rain_model_meta.json`
- `results/pr_curve.png`, `results/roc_curve.png`

### Predict from the latest hour

```bash
weather-cli rain --mode recall     # warn more often
weather-cli rain --mode precision  # fewer false alarms
```

Example output:

```
2025-10-26 23:00:00 | P(rain ≤6h)=0.492 | mode=recall    thr=0.35 → RAIN
2025-10-26 23:00:00 | P(rain ≤6h)=0.492 | mode=precision thr=0.65 → No rain
```

### How thresholds are chosen

Training sweeps precision–recall trade-offs and stores two operating points:

| Threshold type | Purpose                        |
| -------------- | ------------------------------ |
| High recall    | Catch >80 % of rain events     |
| High precision | Warn only when ≥90 % confident |

![PR Curve](results/pr_curve.png)
![ROC Curve](results/roc_curve.png)

### Model Interpretability

ML is not useful unless we can understand what it learned. This section explains why the classifier predicts rain, and not just whether it predicts rain.

- **Feature Coefficients (standardized):** which signals push toward rain vs no-rain

  ```bash
  python scripts/coef_rain.py  # writes top weights
  ```

  Output → `results/coef_top15.txt`

- **Permutation importance:** which features matter most to F1 on the test set.
  This tells us which variables the model relies on the most when making real predictions.
  ```bash
  python scripts/feature_importance_rain.py
  ```
  Output → `results/feature_importance.png`

It engineers both raw signals and short-term deltas/rolling means. Positive coefficients push toward “RAIN”, negative toward “No rain”.

### What the model actually learned (top signals)

| Feature      | Meaning                                                              |
| ------------ | -------------------------------------------------------------------- |
| `precip_mm`  | Existing rainfall strongly predicts more rain (tropical persistence) |
| `temp_c`     | Warmer air holds more moisture → higher chance of near-term rain     |
| `humidity`   | High saturation = cloud condensation is likely                       |
| `pressure`   | Falling pressure indicates unstable atmosphere / storm formation     |
| `cloudcover` | More clouds = conditions building toward rainfall                    |
| `wind_speed` | Negative weight — stronger winds can disperse moisture               |

The classifier isn’t guessing; it’s surfacing familiar meteorological patterns.

### What drives the rain predictions?

Using SHAP explainability, I found that the model mainly relies on **humidity** and **temperature** when deciding if it will rain in the next 12 hours.

- High humidity pushes the model strongly toward predicting rain.
- Lower temperatures slightly increase rain probability.
- The interaction between humidity and temperature mimics real-world weather dynamics — humid, cool conditions tend to precede rainfall.

This means the model isn’t just memorizing data — it has captured meaningful relationships that align with atmospheric science.

![Humidity vs Temperature SHAP interaction](results/shap_interaction.png)

> Generated via `python scripts/explain_shap_interaction.py`, which also writes `results/shap_interaction_rev.png` for the reverse view.

## 🌧️ Rain Events (≥1.0 mm in next 12h)

**Label:** “Rain event if cumulative precipitation ≥ **1.0 mm** within the next **12 hours**.”  
**Policy:** Default to **Early Warning** (recall-leaning) for Lagos conditions. Offer a stricter **Cautious Alert** mode.

**Train / thresholds / predict**

```bash
# (data) pull 90 days of hourly data
make hourly PAST_DAYS=90

# (model) train XGBoost + Isotonic calibration
python scripts/train_xgb_12h_calibrated.py

# (CLI) two operating modes
weather-cli rain --mode recall     # Early Warning (higher recall)
weather-cli rain --mode precision  # Cautious Alert (stricter)
weather-cli rain                   # Balanced (best F1)
```

### 🌧️ Rain Warning (next 12h)
Train tuned model + set guarded thresholds:

```bash
python scripts/xgb_tune_timeseries.py
python scripts/train_xgb_tuned_final.py
cp models/rain_xgb_tuned.joblib    models/rain_classifier_hourly.joblib
cp models/rain_xgb_tuned_meta.json models/rain_model_meta.json
```

## Run Locally

Clone and run:

```bash
make all
```

## CLI

Install (editable):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Once installed, run `weather-cli --help` for all commands (including the rain mode above).
