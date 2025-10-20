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

## Rain Warning (next 6h)

| Mode           | Threshold | Precision | Recall |
| -------------- | --------- | --------- | ------ |
| Default        | 0.50      | 0.71      | 0.70   |
| High-Recall    | 0.35      | 0.68      | 0.84   |
| High-Precision | 0.65      | 0.79      | 0.50   |

Generate data and forecasts:

```bash
make hourly
make rain-train
make rain-now
```

```bash
python scripts/plot_pr_roc.py   # refresh PR/ROC charts
```

![PR Curve](results/pr_curve.png)
![ROC Curve](results/roc_curve.png)

## Run Locally

Clone and run:

```bash
make all
```

## CLI

Install (editable):

## Rain Warning (next 6 hours)

Train once, then predict with two modes:

````bash
make hourly               # fetch + export hourly data
python scripts/train_rain_dual_thresholds.py  # trains & saves thresholds

# Live prediction from latest hour
weather-cli rain --mode recall     # warn more (higher recall)
weather-cli rain --mode precision  # be certain (higher precision)


```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
````
