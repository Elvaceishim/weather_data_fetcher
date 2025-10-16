import json, os, sys
IN, OUT_DIR = "data/weather.json", "results"
OUT = os.path.join(OUT_DIR, "summary.txt")
os.makedirs(OUT_DIR, exist_ok=True)

try:
    with open(IN) as f:
        data = json.load(f)
except FileNotFoundError:
    print("data/weather.json missing. Run `make download`.", file=sys.stderr)
    sys.exit(1)

try:
    daily = data["daily"]
    days = daily["time"]
    tmax = daily["temperature_2m_max"]
    tmin = daily["temperature_2m_min"]
    prec = daily.get("precipitation_sum", [0]*len(days))
except Exception as e:
    print(f"Unexpected JSON structure: {e}", file=sys.stderr)
    sys.exit(2)

with open(OUT, "w") as f:
    f.write("Lagos (Africa/Lagos) – Daily summary\n")
    f.write("-----------------------------------\n")
    for d, lo, hi, p in zip(days, tmin, tmax, prec):
        f.write(f"{d}: {lo}°C – {hi}°C | precip: {p} mm\n")

print(f"✅ Wrote {OUT}")
