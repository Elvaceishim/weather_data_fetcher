"""
FastAPI application exposing the rain nowcast API and a Gradio UI.

The previous Streamlit proxy was difficult to keep alive on Spaces due to
websocket restrictions.  This module provides the same REST endpoints while
mounting a lightweight Gradio front-end so the UI works without websocket
tunnelling.
"""

from __future__ import annotations

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
import gradio as gr
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

# --------- Paths ---------
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SCRIPTS = ROOT / "scripts"

MODEL_PATH = MODELS / "rain_xgb_tuned.joblib"
META_PATH = MODELS / "rain_xgb_tuned_meta.json"
MODEL_JSON_PATH = MODELS / "xgb_tuned.json"
HOURLY_CSV = RESULTS / "hourly.csv"

# Make training utilities importable.
import sys

sys.path.insert(0, str(ROOT))
from scripts.train_xgb_tuned_final import build_features  # type: ignore

# --------- Load model + meta at startup ---------
if not META_PATH.exists():
    raise RuntimeError(
        "Model metadata missing. Run `python scripts/train_xgb_tuned_final.py` "
        "or copy models/rain_xgb_tuned_meta.json into place."
    )

meta = json.loads(META_PATH.read_text())
FEATURES = meta["features"]
THRESH = meta["thresholds"]
HORIZON_H = int(meta["horizon_hours"])


def _load_model() -> XGBClassifier:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    if MODEL_JSON_PATH.exists():
        params = meta.get("model", {}).get("params", {})
        booster = XGBClassifier(**params)
        booster.load_model(MODEL_JSON_PATH)
        return booster

    raise RuntimeError(
        "Model artifact missing. Run `python scripts/train_xgb_tuned_final.py` "
        "to generate models/rain_xgb_tuned.joblib (or xgb_tuned.json), "
        "or copy the trained file into the models/ directory."
    )


model = _load_model()

# --------- Helpers ---------
def ensure_hourly(lat: float, lon: float, past_days: int = 90) -> pd.DataFrame:
    """Refresh the cached hourly CSV when it is missing or stale."""
    env = os.environ.copy()
    env["LAT"] = str(lat)
    env["LON"] = str(lon)
    env["PAST_DAYS"] = str(past_days)

    needs_refresh = True
    if HOURLY_CSV.exists():
        age_hours = (datetime.now().timestamp() - HOURLY_CSV.stat().st_mtime) / 3600
        needs_refresh = age_hours > 6

    if (not HOURLY_CSV.exists()) or needs_refresh:
        try:
            subprocess.run(["bash", str(SCRIPTS / "fetch_weather.sh")], check=True, env=env)
            subprocess.run(["python3", str(SCRIPTS / "export_hourly.py")], check=True, env=env)
        except subprocess.CalledProcessError as exc:
            raise HTTPException(status_code=502, detail=f"Data refresh failed: {exc}") from exc

    return pd.read_csv(HOURLY_CSV, parse_dates=["time"])


def predict_latest(df: pd.DataFrame, mode: str) -> Dict[str, object]:
    """Build features, score the latest hour, and return a structured response."""
    Xdf = build_features(df.copy())
    if Xdf.empty:
        raise HTTPException(status_code=422, detail="Not enough rows to build features.")

    try:
        Xdf = Xdf[FEATURES]
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Feature mismatch: {exc}") from exc

    x = Xdf.iloc[[-1]].values
    probability = float(model.predict_proba(x)[0, 1])

    thresholds = {
        "default": float(THRESH["default"]),
        "recall": float(THRESH["high_recall"]),
        "precision": float(THRESH["high_precision"]),
    }
    if mode not in thresholds:
        raise HTTPException(status_code=400, detail=f"Unsupported mode '{mode}'.")

    threshold = thresholds[mode]
    decision = "RAIN" if probability >= threshold else "No rain"
    ts = df.loc[Xdf.index, "time"].iloc[-1]

    return {
        "timestamp": ts.isoformat(),
        "probability": probability,
        "threshold": threshold,
        "mode": mode,
        "decision": decision,
        "horizon_hours": HORIZON_H,
    }


def format_prediction(result: Dict[str, object]) -> str:
    """Generate a concise markdown summary for the UI."""
    emoji = "🌧️" if result["decision"] == "RAIN" else "✅"
    probability = result["probability"]
    threshold = result["threshold"]
    mode = result["mode"]
    timestamp = result["timestamp"]
    return (
        f"{emoji} **Decision:** {result['decision']} (mode **{mode}**)\n\n"
        f"- Probability of rain ≤ {HORIZON_H}h: **{probability:.3f}**\n"
        f"- Threshold: **{threshold:.2f}**\n"
        f"- Issued for hour ending **{timestamp}**"
    )


class PredictBody(BaseModel):
    lat: float = Field(6.5244, description="Latitude")
    lon: float = Field(3.3792, description="Longitude")
    mode: str = Field("default", description="default | recall | precision")
    past_days: int = Field(90, ge=14, le=180, description="How much history to fetch (days)")


app = FastAPI(title="Rain Nowcast API", version="1.1.0")


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "model_file": MODEL_PATH.name,
        "horizon_hours": HORIZON_H,
        "thresholds": THRESH,
        "features": FEATURES,
    }


@app.post("/predict")
def predict(body: PredictBody) -> Dict[str, object]:
    df = ensure_hourly(body.lat, body.lon, body.past_days)
    out = predict_latest(df, body.mode)
    return {"ok": True, "result": out}


@app.get("/predict")
def predict_get(
    lat: float = Query(6.5244),
    lon: float = Query(3.3792),
    mode: str = Query("default"),
    past_days: int = Query(90, ge=14, le=180),
) -> Dict[str, object]:
    df = ensure_hourly(lat, lon, past_days)
    out = predict_latest(df, mode)
    return {"ok": True, "result": out}


# --------- Gradio UI ---------
CITY_PRESETS: Dict[str, Tuple[float, float]] = {
    "Lagos 🇳🇬": (6.5244, 3.3792),
    "Accra 🇬🇭": (5.6037, -0.1870),
    "Nairobi 🇰🇪": (-1.2864, 36.8172),
    "Kampala 🇺🇬": (0.3476, 32.5825),
    "Addis Ababa 🇪🇹": (8.9806, 38.7578),
    "Custom": (0.0, 0.0),
}


def _resolve_location(city: str, lat: float, lon: float) -> Tuple[float, float, str]:
    if city in CITY_PRESETS and city != "Custom":
        chosen_lat, chosen_lon = CITY_PRESETS[city]
        label = city
    else:
        chosen_lat, chosen_lon = lat, lon
        label = f"Custom ({lat:.3f}, {lon:.3f})"
    return chosen_lat, chosen_lon, label


def gradio_predict(
    city: str,
    lat: float,
    lon: float,
    mode: str,
    past_days: int,
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    chosen_lat, chosen_lon, label = _resolve_location(city, lat, lon)
    df = ensure_hourly(chosen_lat, chosen_lon, past_days)
    result = predict_latest(df, mode)

    summary = format_prediction(result)

    last48 = df.tail(48).copy()
    last48.set_index("time", inplace=True)
    chart = last48[["temp_c", "humidity", "precip_mm", "rain_mm"]]

    latest = pd.DataFrame(
        {
            "location": [label],
            "timestamp": [result["timestamp"]],
            "mode": [result["mode"]],
            "probability": [result["probability"]],
            "threshold": [result["threshold"]],
            "decision": [result["decision"]],
        }
    )
    return summary, latest, chart


with gr.Blocks(css=".gradio-container {max-width: 900px;}") as demo:
    gr.Markdown("# 🌧️ Rain Nowcast\nPredict the probability of rain in the next "
                f"{HORIZON_H} hours using the tuned XGBoost model.")

    with gr.Row():
        city_input = gr.Dropdown(
            label="City preset",
            choices=list(CITY_PRESETS.keys()),
            value="Lagos 🇳🇬",
        )
        mode_input = gr.Radio(
            label="Decision mode",
            choices=["default", "recall", "precision"],
            value="default",
            info="default=balanced, recall=warn more, precision=extra picky",
        )

    with gr.Row():
        lat_input = gr.Number(label="Latitude (used if city is Custom)", value=6.5244)
        lon_input = gr.Number(label="Longitude (used if city is Custom)", value=3.3792)
        past_days_input = gr.Slider(
            label="History window (days)",
            minimum=14,
            maximum=180,
            value=90,
            step=1,
        )

    submit = gr.Button("Run prediction", variant="primary")

    summary_md = gr.Markdown()
    latest_df = gr.Dataframe(label="Latest prediction", wrap=True)
    chart_df = gr.LinePlot(
        label="Last 48h weather (hourly)",
        x="time",
        y=["temp_c", "humidity", "precip_mm", "rain_mm"],
        overlay_point=True,
        width="100%",
        height=350,
    )

    submit.click(
        gradio_predict,
        inputs=[city_input, lat_input, lon_input, mode_input, past_days_input],
        outputs=[summary_md, latest_df, chart_df],
    )

    gr.Markdown(
        "Model features match the training pipeline "
        "(see `scripts/train_xgb_tuned_final.py`). Data fetched from Open-Meteo."
    )


app = gr.mount_gradio_app(app, demo, path="/")
