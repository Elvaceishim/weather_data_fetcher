
from pathlib import Path
from datetime import datetime
import os, json, subprocess

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

# --------- Paths ---------
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SCRIPTS = ROOT / "scripts"

MODEL_PATH = MODELS / "rain_xgb_tuned.joblib"
META_PATH  = MODELS / "rain_xgb_tuned_meta.json"
HOURLY_CSV = RESULTS / "hourly.csv"

import sys
sys.path.insert(0, str(ROOT))
from scripts.train_xgb_tuned_final import build_features  # reuse your exact feature builder

# --------- Load model + meta at startup ---------
model = joblib.load(MODEL_PATH)
meta = json.loads(META_PATH.read_text())
FEATURES = meta["features"]
THRESH = meta["thresholds"]
HORIZON_H = int(meta["horizon_hours"])

app = FastAPI(title="Rain Nowcast API", version="1.0.0")

# --------- Helpers ---------
def ensure_hourly(lat: float, lon: float, past_days: int = 90) -> pd.DataFrame:
    env = os.environ.copy()
    env["LAT"] = str(lat); env["LON"] = str(lon); env["PAST_DAYS"] = str(past_days)
    need_refresh = True
    if HOURLY_CSV.exists():
        age_hours = (datetime.now().timestamp() - HOURLY_CSV.stat().st_mtime) / 3600
        need_refresh = age_hours > 6
    if (not HOURLY_CSV.exists()) or need_refresh:
        subprocess.run(["bash", str(SCRIPTS / "fetch_weather.sh")], check=True, env=env)
        subprocess.run(["python3", str(SCRIPTS / "export_hourly.py")], check=True, env=env)
    return pd.read_csv(HOURLY_CSV, parse_dates=["time"])

def predict_latest(df: pd.DataFrame, mode: str):
    Xdf = build_features(df.copy())
    Xdf = Xdf[FEATURES]
    x = Xdf.iloc[[-1]].values
    p = float(model.predict_proba(x)[0, 1])
    thr_map = {
        "default":   float(THRESH["default"]),
        "recall":    float(THRESH["high_recall"]),
        "precision": float(THRESH["high_precision"]),
    }
    t = thr_map[mode]
    decision = "RAIN" if p >= t else "No rain"
    ts = df.loc[Xdf.index, "time"].iloc[-1]
    return dict(
        timestamp=str(ts),
        probability=p,
        threshold=t,
        mode=mode,
        decision=decision,
        horizon_hours=HORIZON_H,
    )

# --------- Schemas ---------
class PredictBody(BaseModel):
    lat: float = Field(6.5244, description="Latitude")
    lon: float = Field(3.3792, description="Longitude")
    mode: str = Field("default", description="default | recall | precision")
    past_days: int = Field(90, ge=14, le=180, description="How much history to fetch (days)")

# --------- Endpoints ---------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_file": MODEL_PATH.name,
        "horizon_hours": HORIZON_H,
        "thresholds": THRESH,
        "features": FEATURES,
    }

@app.post("/predict")
def predict(body: PredictBody):
    df = ensure_hourly(body.lat, body.lon, body.past_days)
    out = predict_latest(df, body.mode)
    return {"ok": True, "result": out}

@app.get("/predict")
def predict_get(
    lat: float = Query(6.5244), lon: float = Query(3.3792),
    mode: str = Query("default"), past_days: int = Query(90)
):
    df = ensure_hourly(lat, lon, past_days)
    out = predict_latest(df, mode)
    return {"ok": True, "result": out}
