import json
import os
from shutil import copyfile


RESULTS_DIR = "results"
ASSETS_DIR = "assets"
TEMPS_RESULTS = os.path.join(RESULTS_DIR, "temps.png")
PRECIP_RESULTS = os.path.join(RESULTS_DIR, "precip.png")
TEMPS_ASSET = os.path.join(ASSETS_DIR, "temps.png")
PRECIP_ASSET = os.path.join(ASSETS_DIR, "precip.png")

os.environ.setdefault("MPLCONFIGDIR", os.path.join(RESULTS_DIR, ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(RESULTS_DIR, ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)


def mirror_asset(src: str, dest: str) -> None:
    copyfile(src, dest)


def main():
    with open("data/weather.json") as handle:
        data = json.load(handle)

    days = pd.to_datetime(data["daily"]["time"])
    tmax = pd.Series(data["daily"]["temperature_2m_max"])
    tmin = pd.Series(data["daily"]["temperature_2m_min"])
    prec = pd.Series(data["daily"].get("precipitation_sum", [0] * len(days)))

    ensure_dirs()

    # Temperature line chart
    plt.figure()
    plt.plot(days, tmax, marker="o", label="Max °C")
    plt.plot(days, tmin, marker="o", label="Min °C")
    plt.xticks(rotation=45, ha="right")
    plt.title("Daily Temperatures (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(TEMPS_RESULTS)
    plt.close()

    mirror_asset(TEMPS_RESULTS, TEMPS_ASSET)

    # Precipitation bar chart
    plt.figure()
    plt.bar(days, prec)
    plt.xticks(rotation=45, ha="right")
    plt.title("Daily Precipitation (mm)")
    plt.tight_layout()
    plt.savefig(PRECIP_RESULTS)
    plt.close()

    mirror_asset(PRECIP_RESULTS, PRECIP_ASSET)

    print(f"✅ Wrote {TEMPS_RESULTS} / {PRECIP_RESULTS}")
    print(f"✅ Updated assets at {TEMPS_ASSET} / {PRECIP_ASSET}")

if __name__ == "__main__":
    main()
