#!/usr/bin/env python3
import json, os, argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd, numpy as np, subprocess

META  = Path("models/rain_xgb_tuned_meta.json")
LOGS  = Path("logs")
PRED_LOG = LOGS / "predictions.csv"

def ensure_hourly(lat, lon, past_days=120):
    env = os.environ.copy()
    env["LAT"], env["LON"], env["PAST_DAYS"] = str(lat), str(lon), str(past_days)
    subprocess.run(["bash", "scripts/fetch_weather.sh"], check=True, env=env)
    subprocess.run(["python3", "scripts/export_hourly.py"], check=True, env=env)
    return pd.read_csv("results/hourly.csv", parse_dates=["time"])

def label_from_df(df, ts_pred, horizon_h, event_mm):
    # find the row with time == ts_pred, then sum next H hours of precip_mm
    # allow slight mismatch by nearest timestamp within 1 hour
    idx = (df["time"] - ts_pred).abs().idxmin()
    if abs((df.loc[idx, "time"] - ts_pred).total_seconds()) > 3600:
        return None  # can't align
    end_idx = min(idx + horizon_h, len(df)-1)
    total = float(np.nansum(df.loc[idx+1:end_idx, "precip_mm"]))
    return 1 if total >= event_mm else 0

def main():
    if not PRED_LOG.exists():
        print("No predictions.csv found.")
        return

    meta = json.loads(Path(META).read_text())
    H = int(meta["horizon_hours"]); event_mm = float(meta["event_mm"])

    df = pd.read_csv(PRED_LOG, parse_dates=["ts_pred","logged_at"])
    updated = 0
    for i, row in df[df["y_true"].isna() | (df["y_true"]=="")].iterrows():
        ts_pred = row["ts_pred"]
        if datetime.now() < ts_pred + timedelta(hours=H):
            continue  # horizon not passed yet
        # fetch enough history to cover that timestamp
        hdf = ensure_hourly(row["lat"], row["lon"], past_days=120)
        y = label_from_df(hdf, ts_pred, H, event_mm)
        if y is not None:
            df.at[i, "y_true"] = int(y)
            updated += 1

    df.to_csv(PRED_LOG, index=False)
    print(f"Backfilled {updated} rows into {PRED_LOG}")

if __name__ == "__main__":
    main()
