#!/usr/bin/env python3
import argparse
import os
import json
import joblib
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path

def fetch_weather(lat, lon, city, out_json):
    import requests, json as _json
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        "&timezone=Africa%2FLagos"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    with open(out_json, "w") as f:
        _json.dump(r.json(), f)

def process_weather(in_json, out_csv):
    import json as _json
    data = _json.load(open(in_json))
    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "temp_min_c": data["daily"]["temperature_2m_min"],
        "temp_max_c": data["daily"]["temperature_2m_max"],
        "precip_mm": data["daily"].get("precipitation_sum", [0]*len(data["daily"]["time"]))
    })
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/weather_cli.csv", index=False)
    return df

def plot_weather(df, city):
    import matplotlib.pyplot as plt
    days = pd.to_datetime(df["date"])
    plt.figure()
    plt.plot(days, df["temp_max_c"], marker="o", label="Max °C")
    plt.plot(days, df["temp_min_c"], marker="o", label="Min °C")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{city} — Daily Temperatures")
    plt.legend()
    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/cli_plot.png")
    plt.close()

def ensure_hourly_data(lat, lon, past_days=14):
    """
    Make sure results/hourly.csv exists.
    If not, call your existing fetch + export scripts so we don’t duplicate logic.
    """
    hourly_csv = Path("results/hourly.csv")
    if hourly_csv.exists():
        return hourly_csv

    env = os.environ.copy()
    env["LAT"] = str(lat)
    env["LON"] = str(lon)
    env["PAST_DAYS"] = str(past_days)

    print(f"[cli] Generating hourly data for LAT={lat} LON={lon} PAST_DAYS={past_days} …")
    subprocess.run(["bash", "scripts/fetch_weather.sh"], check=True, env=env)
    subprocess.run(["python3", "scripts/export_hourly.py"], check=True, env=env)

    if not hourly_csv.exists():
        raise FileNotFoundError("results/hourly.csv not found after generation.")
    return hourly_csv

def rebuild_features_like_training(df: pd.DataFrame, features_from_meta: list) -> pd.DataFrame:
    """
    Rebuild the exact feature columns used during training:
    base + d_ (1h diff) + ma3_ (3h rolling mean).
    """
    base = ["temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"]

    for col in base:
        df[f"d_{col}"] = df[col].diff()
        df[f"ma3_{col}"] = df[col].rolling(3).mean()
    df = df.dropna().reset_index(drop=True)
    return df[features_from_meta]

def cmd_rain(args):
    meta_path = Path("models/rain_model_meta.json")
    model_path = Path("models/rain_classifier_hourly.joblib")
    if not (meta_path.exists() and model_path.exists()):
        raise FileNotFoundError(
            "Rain model files not found. Train them first:\n"
            "  python scripts/train_rain_dual_thresholds.py"
        )
    meta = json.load(open(meta_path))
    clf = joblib.load(model_path)

    hourly_csv = ensure_hourly_data(args.lat, args.lon, past_days=args.past_days)
    df = pd.read_csv(hourly_csv, parse_dates=["time"])

    feat_df = rebuild_features_like_training(df.copy(), meta["features"])
    X = feat_df.iloc[[-1]].values
    p = float(clf.predict_proba(X)[0, 1])

    thr_map = {
        "default": meta["thresholds"]["default"],
        "recall": meta["thresholds"]["high_recall"],
        "precision": meta["thresholds"]["high_precision"],
    }
    thr = float(thr_map[args.mode])
    decision = "RAIN" if p >= thr else "No rain"

    ts = df["time"].iloc[-1]
    print(f"{ts}  |  P(rain ≤{meta['horizon_hours']}h)={p:.3f}  |  mode={args.mode} thr={thr:.2f}  →  {decision}")

def main():
    parser = argparse.ArgumentParser(prog="weather-cli", description="Weather pipeline + rain warning")
    sub = parser.add_subparsers(dest="cmd")

    parser.add_argument("--city", default="Lagos")
    parser.add_argument("--lat", type=float, default=6.5244)
    parser.add_argument("--lon", type=float, default=3.3792)

    p_rain = sub.add_parser("rain", help="Rain warning for next 6h (dual thresholds)")
    p_rain.add_argument("--mode", choices=["recall","precision","default"], default="recall")
    p_rain.add_argument("--lat", type=float, default=6.5244)
    p_rain.add_argument("--lon", type=float, default=3.3792)
    p_rain.add_argument("--past_days", type=int, default=14)
    p_rain.set_defaults(func=cmd_rain)

    args = parser.parse_args()

    if getattr(args, "cmd", None) == "rain":
        return args.func(args)

    Path("data").mkdir(exist_ok=True)
    out_json = "data/weather_cli.json"
    df = None
    try:
        fetch_weather(args.lat, args.lon, args.city, out_json)
        df = process_weather(out_json, "results/weather_cli.csv")
        plot_weather(df, args.city)
        print("✅ Daily pipeline complete. See results/cli_plot.png")
    except Exception as e:
        print(f"❌ Pipeline error: {e}")

if __name__ == "__main__":
    main()
