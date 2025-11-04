#!/usr/bin/env python3
import argparse, os, json
from pathlib import Path
from datetime import datetime
import joblib, pandas as pd, numpy as np, subprocess

MODEL = Path("models/rain_xgb_tuned.joblib")
META  = Path("models/rain_xgb_tuned_meta.json")
HOURLY = Path("results/hourly.csv")
LOGS = Path("logs"); LOGS.mkdir(exist_ok=True)
PRED_LOG = LOGS / "predictions.csv"

def ensure_hourly(lat, lon, past_days=90):
    env = os.environ.copy()
    env["LAT"], env["LON"], env["PAST_DAYS"] = str(lat), str(lon), str(past_days)
    if (not HOURLY.exists()):
        subprocess.run(["bash", "scripts/fetch_weather.sh"], check=True, env=env)
        subprocess.run(["python3", "scripts/export_hourly.py"], check=True, env=env)
    return pd.read_csv(HOURLY, parse_dates=["time"])

def build_features_like_training(df, features):
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_xgb_tuned_final", "scripts/train_xgb_tuned_final.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_features = module.build_features
    Xdf = build_features(df)
    return Xdf[features]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Lagos")
    ap.add_argument("--lat", type=float, default=6.5244)
    ap.add_argument("--lon", type=float, default=3.3792)
    ap.add_argument("--mode", choices=["default","recall","precision"], default="default")
    args = ap.parse_args()

    meta = json.loads(META.read_text())
    thr = meta["thresholds"]; feats = meta["features"]; H = meta["horizon_hours"]; event_mm = meta["event_mm"]

    df = ensure_hourly(args.lat, args.lon, 90)
    Xdf = build_features_like_training(df.copy(), feats)
    if Xdf.empty: raise SystemExit("Not enough rows to build features")

    clf = joblib.load(MODEL)
    p = float(clf.predict_proba(Xdf.iloc[[-1]].values)[0,1])
    tmap = {"default":thr["default"], "recall":thr["high_recall"], "precision":thr["high_precision"]}
    t = float(tmap[args.mode])
    decision = "RAIN" if p >= t else "No rain"

    row = {
        "ts_pred": df.loc[Xdf.index, "time"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city": args.city, "lat": args.lat, "lon": args.lon,
        "mode": args.mode, "horizon_h": H, "event_mm": event_mm,
        "p": p, "threshold": t, "decision": decision,
        "y_true": "",  # to be filled by backfill
    }
    if not PRED_LOG.exists():
        pd.DataFrame([row]).to_csv(PRED_LOG, index=False)
    else:
        pd.DataFrame([row]).to_csv(PRED_LOG, mode="a", header=False, index=False)

    print(f"Logged: {row}")

if __name__ == "__main__":
    main()
