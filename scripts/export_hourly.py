import os
import sys
import json
import pandas as pd

os.makedirs("results", exist_ok=True)
with open("data/weather.json") as handle:
    data = json.load(handle)

if "hourly" not in data:
    print(
        "data/weather.json missing 'hourly'. Re-run the fetch step with hourly "
        "parameters enabled (see scripts/fetch_weather.sh).",
        file=sys.stderr,
    )
    sys.exit(1)

H = data["hourly"]
df = pd.DataFrame({
    "time": H["time"],
    "temp_c": H["temperature_2m"],
    "humidity": H["relative_humidity_2m"],
    "cloudcover": H["cloudcover"],
    "pressure": H["pressure_msl"],
    "wind_speed": H["wind_speed_10m"],
    "precip_mm": H["precipitation"],
    "rain_mm": H["rain"],
})

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)
df.to_csv("results/hourly.csv", index=False)
print(f"✅ Wrote results/hourly.csv with {len(df)} rows")
