
import json, pandas as pd, os

os.makedirs("results", exist_ok=True)
data = json.load(open("data/weather.json"))

df = pd.DataFrame({
    "date": data["daily"]["time"],
    "temp_min_c": data["daily"]["temperature_2m_min"],
    "temp_max_c": data["daily"]["temperature_2m_max"],
    "precip_mm": data["daily"]["precipitation_sum"],
    "cloudcover": data["daily"]["cloudcover_mean"],
    "wind_speed": data["daily"]["wind_speed_10m_max"],
    "humidity_max": data["daily"]["relative_humidity_2m_max"],
    "humidity_min": data["daily"]["relative_humidity_2m_min"],
})
df.to_csv("results/daily.csv", index=False)
print(f"✅ Wrote results/daily.csv with {len(df)} rows")
