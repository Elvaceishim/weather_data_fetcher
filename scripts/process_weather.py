from dotenv import load_dotenv
import os, json, sys, logging

load_dotenv() 
LAT = os.getenv("LAT", "6.5244")
LON = os.getenv("LON", "3.3792")
CITY = os.getenv("CITY", "Lagos")
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logging.info(f"Processing weather for {CITY} ({LAT}, {LON})")

IN, OUT_DIR = "data/weather.json", "results"
OUT = os.path.join(OUT_DIR, "summary.txt")
os.makedirs(OUT_DIR, exist_ok=True)

logging.info("Reading weather.json")

try:
    with open(IN) as f:
        data = json.load(f)
except FileNotFoundError:
    print("weather.json not found. Run `make download`.", file=sys.stderr); sys.exit(1)

try:
    daily = data["daily"]
    days = daily["time"]
    tmax = daily["temperature_2m_max"]
    tmin = daily["temperature_2m_min"]
    prec = daily.get("precipitation_sum", [0]*len(days))
except Exception as e:
    print(f"Unexpected JSON structure: {e}", file=sys.stderr); sys.exit(2)

with open(OUT, "w") as f:
    f.write("Lagos (Africa/Lagos) – Daily summary\n")
    f.write("-----------------------------------\n")
    for d, lo, hi, p in zip(days, tmin, tmax, prec):
        f.write(f"{d}: {lo}°C – {hi}°C | precip: {p} mm\n")

logging.info(f"Wrote summary to {OUT}")

print(f"✅ Wrote {OUT}")

import csv
with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["date", "temp_min_c", "temp_max_c", "precip_mm"])
    for d, lo, hi, p in zip(days, tmin, tmax, prec):
        w.writerow([d, lo, hi, p])
print("✅ Wrote results/summary.csv")
