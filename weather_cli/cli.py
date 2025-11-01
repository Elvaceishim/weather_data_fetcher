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
    Rebuild EXACT feature columns used during training.
    Assumes df has hourly columns: time, temp_c, humidity, cloudcover, pressure,
    wind_speed, precip_mm, rain_mm.
    Returns a DataFrame with columns ordered as in features_from_meta.
    """
    required = {"time","temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hourly data missing columns: {sorted(missing)}")

    # Base + short-term dynamics
    base = ["temp_c","humidity","cloudcover","pressure","wind_speed","precip_mm","rain_mm"]
    for c in base:
        df[f"d_{c}"]   = df[c].diff()
        df[f"ma3_{c}"] = df[c].rolling(3).mean()

    # Onset (3h deltas)
    for c in ["pressure","humidity","cloudcover","temp_c"]:
        df[f"d3_{c}"] = df[c] - df[c].shift(3)

    # Dewpoint proxy + dynamics
    df["dew_proxy"]     = df["temp_c"] - (df["humidity"] / 5.0)
    df["d_dew_proxy"]   = df["dew_proxy"].diff()
    df["ma3_dew_proxy"] = df["dew_proxy"].rolling(3).mean()

    # Intensity & persistence (past-only)
    df["rain_sum_3h"]   = df["precip_mm"].rolling(3).sum()
    df["rain_sum_6h"]   = df["precip_mm"].rolling(6).sum()
    df["rain_sum_12h"]  = df["precip_mm"].rolling(12).sum()
    df["rain_sum_24h"]  = df["precip_mm"].rolling(24).sum()
    df["rain_max_6h"]   = df["precip_mm"].rolling(6).max()
    df["rain_max_12h"]  = df["precip_mm"].rolling(12).max()

    # Wet/dry streaks (hours)
    is_raining = (df["precip_mm"] > 0).astype(int)
    dry = (~(is_raining.astype(bool))).astype(int)
    df["dry_streak_h"] = (dry.groupby((dry != dry.shift()).cumsum()).cumcount() + 1) * dry
    df["dry_streak_h"] = df["dry_streak_h"].where(dry == 1, 0)

    wet = is_raining
    df["wet_streak_h"] = (wet.groupby((wet != wet.shift()).cumsum()).cumcount() + 1) * wet
    df["wet_streak_h"] = df["wet_streak_h"].where(wet == 1, 0)

    # Cycles (diurnal + weekly + seasonal hour-of-year)
    df["hour"] = df["time"].dt.hour
    df["dow"]  = df["time"].dt.dayofweek
    df["doy"]  = df["time"].dt.dayofyear
    df["hoy"]  = (df["doy"] - 1) * 24 + df["hour"]

    # sin/cos encodings
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7.0)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7.0)
    df["hoy_sin"]  = np.sin(2*np.pi*df["hoy"]/(365.25*24))
    df["hoy_cos"]  = np.cos(2*np.pi*df["hoy"]/(365.25*24))

    # Light interactions (help precision)
    df["hum_x_cloud"]   = df["humidity"] * df["cloudcover"]
    df["wind_x_cloud"]  = df["wind_speed"] * df["cloudcover"]
    df["press_drop_3h"] = -df["d3_pressure"]  # pressure falling → storms
    df["press_drop_6h"] = df["pressure"].shift(6) - df["pressure"]

    # We need enough history for 24h windows. Warn if tiny.
    if len(df) < 24:
        raise ValueError("Not enough hourly rows to build 24h features. Fetch more history (PAST_DAYS≥2).")

    # Align with training (drop rows with NaNs from rolling/diff)
    df = df.dropna().reset_index(drop=True)

    # Final column order: exactly as training meta recorded
    missing_feats = [c for c in features_from_meta if c not in df.columns]
    if missing_feats:
        raise ValueError(f"CLI feature builder missing columns expected by model: {missing_feats}")

    return df[features_from_meta]

def cmd_rain(args):
    meta_path = Path("models/rain_model_meta.json")
    model_path = Path("models/rain_classifier_hourly.joblib")
    if not (meta_path.exists() and model_path.exists()):
        raise FileNotFoundError(
            "Rain model files not found. Train them first:\n"
            "  python scripts/train_xgb_12h_calibrated.py"
        )

    # Load model + metadata
    meta = json.load(open(meta_path))
    clf = joblib.load(model_path)

    # Ensure hourly.csv exists
    hourly_csv = ensure_hourly_data(args.lat, args.lon, past_days=args.past_days)
    df = pd.read_csv(hourly_csv, parse_dates=["time"])

    # Rebuild features exactly as in training
    feat_df = rebuild_features_like_training(df.copy(), meta["features"])
    X = feat_df.iloc[[-1]].values
    p = float(clf.predict_proba(X)[0, 1])

    # Pick threshold based on mode
    thresholds = meta["thresholds"]
    if args.mode == "recall":
        thr = float(thresholds.get("high_recall", thresholds["default"]))
    elif args.mode == "precision":
        thr = float(thresholds.get("high_precision", thresholds["default"]))
    else:
        thr = float(thresholds["default"])

    # Interpret
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
