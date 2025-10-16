import json
import os
import sys
from pathlib import Path


DATA_PATH = Path("data/weather.json")
RESULTS_DIR = Path("results")
PLOT_PATH = RESULTS_DIR / "weather_plot.png"


def load_daily_weather():
    if not DATA_PATH.exists():
        print("data/weather.json missing. Run `make download` first.", file=sys.stderr)
        sys.exit(1)

    try:
        with DATA_PATH.open() as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        print(f"Failed to parse {DATA_PATH}: {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        daily = payload["daily"]
        days = daily["time"]
        tmax = daily["temperature_2m_max"]
        tmin = daily["temperature_2m_min"]
        precip = daily.get("precipitation_sum", [0.0] * len(days))
    except KeyError as exc:
        print(f"Missing key in weather data: {exc}", file=sys.stderr)
        sys.exit(3)

    if not (len(days) == len(tmax) == len(tmin) == len(precip)):
        print("Inconsistent array lengths in daily weather data.", file=sys.stderr)
        sys.exit(4)

    timezone = payload.get("timezone", "Unknown timezone")
    location = payload.get("latitude"), payload.get("longitude")
    return days, tmax, tmin, precip, timezone, location


def make_plot(days, tmax, tmin, precip, timezone, location):
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(RESULTS_DIR / ".cache"))

    mpl_config_dir = Path(os.environ["MPLCONFIGDIR"])
    mpl_cache_dir = Path(os.environ["XDG_CACHE_HOME"])
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib not installed. Install it with `python3 -m pip install matplotlib`.",
            file=sys.stderr,
        )
        sys.exit(5)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    indices = range(len(days))
    lat, lon = location
    title_parts = ["Daily weather"]
    if lat is not None and lon is not None:
        title_parts.append(f"({lat:.4f}, {lon:.4f})")
    if timezone:
        title_parts.append(f"– {timezone}")
    title = " ".join(title_parts)

    fig, ax_temp = plt.subplots(figsize=(10, 5))
    ax_temp.plot(indices, tmax, marker="o", label="Max temp (°C)")
    ax_temp.plot(indices, tmin, marker="s", label="Min temp (°C)")
    ax_temp.set_xticks(list(indices))
    ax_temp.set_xticklabels(days, rotation=45, ha="right")
    ax_temp.set_ylabel("Temperature (°C)")
    ax_temp.set_title(title)

    ax_precip = ax_temp.twinx()
    ax_precip.bar(indices, precip, alpha=0.3, color="#1f77b4", label="Precipitation (mm)")
    ax_precip.set_ylabel("Precipitation (mm)")

    lines, labels = ax_temp.get_legend_handles_labels()
    bars, bar_labels = ax_precip.get_legend_handles_labels()
    ax_temp.legend(lines + bars, labels + bar_labels, loc="upper left")

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {PLOT_PATH}")


def main():
    days, tmax, tmin, precip, timezone, location = load_daily_weather()
    make_plot(days, tmax, tmin, precip, timezone, location)


if __name__ == "__main__":
    main()
