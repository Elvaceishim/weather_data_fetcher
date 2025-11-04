
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import subprocess
import os
from datetime import datetime, timedelta

# Settings
MODEL_PATH = Path("models/rain_xgb_tuned.joblib")
META_PATH  = Path("models/rain_xgb_tuned_meta.json")
HOURLY_CSV = Path("results/hourly.csv")

# Load model + meta
@st.cache_resource
def load_model():
    if not (MODEL_PATH.exists() and META_PATH.exists()):
        st.error("Trained model not found. Run: python scripts/xgb_tune_timeseries.py && python scripts/train_xgb_tuned_final.py")
        st.stop()
    clf = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return clf, meta

def build_features_like_training(df: pd.DataFrame, features: list) -> pd.DataFrame:
    from scripts.train_xgb_tuned_final import build_features  # reuse your code
    Xdf = build_features(df)
    return Xdf[features]

def ensure_hourly(lat: float, lon: float, past_days: int = 90) -> pd.DataFrame:
    env = os.environ.copy()
    env["LAT"] = str(lat)
    env["LON"] = str(lon)
    env["PAST_DAYS"] = str(past_days)

    # If file is missing or stale (>12h), refresh
    needs_refresh = True
    if HOURLY_CSV.exists():
        age_hours = (datetime.now() - datetime.fromtimestamp(HOURLY_CSV.stat().st_mtime)).total_seconds() / 3600.0
        needs_refresh = age_hours > 12

    if (not HOURLY_CSV.exists()) or needs_refresh:
        st.info("Fetching fresh hourly weather…")
        subprocess.run(["bash", "scripts/fetch_weather.sh"], check=True, env=env)
        subprocess.run(["python3", "scripts/export_hourly.py"], check=True, env=env)

    return pd.read_csv(HOURLY_CSV, parse_dates=["time"])

# UI
st.set_page_config(page_title="Rain Nowcast (12h)", page_icon="🌧️", layout="centered")
st.title("🌧️ Rain Nowcast — next 12 hours")

clf, meta = load_model()
features = meta["features"]
thr = meta["thresholds"]
horizon_h = meta["horizon_hours"]

# Presets for cities
CITY_PRESETS = {
    "Lagos 🇳🇬":   (6.5244, 3.3792),
    "Accra 🇬🇭":   (5.6037, -0.1870),
    "Nairobi 🇰🇪": (-1.2864, 36.8172),
    "Kampala 🇺🇬": (0.3476, 32.5825),
    "Addis 🇪🇹":   (8.9806, 38.7578),
}

col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("City", list(CITY_PRESETS.keys()), index=0)
with col2:
    mode = st.selectbox("Decision mode", ["default", "recall", "precision"], index=0)

lat, lon = CITY_PRESETS[city]
st.caption(f"Lat/Lon: **{lat:.4f}, {lon:.4f}** • Horizon: **{horizon_h}h** • Mode: **{mode}**")

df = ensure_hourly(lat, lon, past_days=90)

Xdf = build_features_like_training(df.copy(), features)
if Xdf.empty:
    st.error("Not enough data to build features. Try again after fetch.")
    st.stop()

x_last = Xdf.iloc[[-1]].values
p = float(clf.predict_proba(x_last)[0, 1])
thr_map = {
    "default":   float(thr["default"]),
    "recall":    float(thr["high_recall"]),
    "precision": float(thr["high_precision"]),
}
t = thr_map[mode]
decision = "RAIN" if p >= t else "No rain"

st.subheader("Prediction")
st.metric(
    label=f"P(rain ≤ {horizon_h}h)",
    value=f"{p:.3f}",
    delta=f"threshold={t:.2f}",
    delta_color="inverse" if p < t else "normal"
)
st.markdown(
    f"**Decision:** {'🌧️ RAIN' if decision=='RAIN' else '✅ No rain'}  "
    f"(mode **{mode}**, threshold **{t:.2f}**)"
)

st.subheader("Last 48h — context")
last48 = df.tail(48).copy()
c1, c2 = st.columns(2)
with c1:
    st.line_chart(data=last48.set_index("time")[["temp_c", "humidity"]])
with c2:
    st.line_chart(data=last48.set_index("time")[["precip_mm", "rain_mm"]])

# --- Probability sparkline over last 48h ---
st.subheader("Last 48h — rain probability")
# Recompute probabilities for all available rows, then show last 48 aligned to time
probas_all = clf.predict_proba(Xdf.values)[:, 1]
proba_series = pd.Series(probas_all, index=Xdf.index, name="p_rain")
# Align times (Xdf is derived from df; both share row order except dropped NaNs at head)
times_aligned = df.loc[Xdf.index, "time"]
last48_p = pd.DataFrame({"time": times_aligned, "p_rain": proba_series}).tail(48).set_index("time")
st.line_chart(last48_p)

# --- Download buttons ---
st.subheader("Downloads")
st.download_button(
    label="⬇️ Download hourly.csv",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="hourly.csv",
    mime="text/csv",
)

latest_frame = pd.DataFrame({
    "time": [df.loc[Xdf.index, "time"].iloc[-1]],
    "p_rain_next_12h": [p],
    "mode": [mode],
    "threshold": [t],
    "decision": [decision],
})
st.download_button(
    label="⬇️ Download latest_prediction.csv",
    data=latest_frame.to_csv(index=False).encode("utf-8"),
    file_name="latest_prediction.csv",
    mime="text/csv",
)

# Explain thresholds
with st.expander("What do these modes mean?"):
    st.write("""
- **default**: balanced (good everyday choice)
- **recall**: warn more (catches more rain, may over-warn)
- **precision**: be picky (alerts are rare but confident)
""")

st.caption("Model: XGBoost (tuned) • Features rebuilt exactly like training • Data: Open-Meteo hourly")
