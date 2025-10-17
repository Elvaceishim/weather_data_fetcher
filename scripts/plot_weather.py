import os
import json
import pandas as pd
import matplotlib.pyplot as plt

with open("data/weather.json") as f:
    data = json.load(f)

days = pd.to_datetime(data["daily"]["time"])
tmax = pd.Series(data["daily"]["temperature_2m_max"])
tmin = pd.Series(data["daily"]["temperature_2m_min"])
prec = pd.Series(data["daily"].get("precipitation_sum", [0] * len(days)))

os.makedirs("results", exist_ok=True)

# Temperature line chart
plt.figure()
plt.plot(days, tmax, marker="o", label="Max °C")
plt.plot(days, tmin, marker="o", label="Min °C")
plt.xticks(rotation=45, ha="right")
plt.title("Daily Temperatures (°C)")
plt.legend()
plt.tight_layout()
plt.savefig("results/temps.png")
plt.close()

# ---- Precipitation bar chart
plt.figure()
plt.bar(days, prec)
plt.xticks(rotation=45, ha="right")
plt.title("Daily Precipitation (mm)")
plt.tight_layout()
plt.savefig("results/precip.png")
plt.close()


print("✅ Wrote results/temps.png and results/precip.png")
