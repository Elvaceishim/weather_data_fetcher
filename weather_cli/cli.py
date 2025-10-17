
import argparse, os, json, requests, pandas as pd, matplotlib.pyplot as plt

def fetch_weather(lat, lon, city, out_json):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        "&timezone=Africa%2FLagos"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    with open(out_json, "w") as f:
        json.dump(r.json(), f)
    print(f"✅ Downloaded weather data for {city}")

def process_weather(in_json, out_csv):
    data = json.load(open(in_json))
    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "temp_min_c": data["daily"]["temperature_2m_min"],
        "temp_max_c": data["daily"]["temperature_2m_max"],
        "precip_mm": data["daily"].get("precipitation_sum", [0]*len(data["daily"]["time"]))
    })
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved processed data → {out_csv}")
    return df

def plot_weather(df, city):
    days = pd.to_datetime(df["date"])
    plt.plot(days, df["temp_max_c"], marker="o", label="Max °C")
    plt.plot(days, df["temp_min_c"], marker="o", label="Min °C")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{city} — Daily Temperatures")
    plt.legend()
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/cli_plot.png")
    print("✅ Wrote results/cli_plot.png")

def main():
    parser = argparse.ArgumentParser(description="Weather CLI Tool")
    parser.add_argument("--city", default="Lagos", help="City name (for label only)")
    parser.add_argument("--lat", type=float, default=6.5244, help="Latitude")
    parser.add_argument("--lon", type=float, default=3.3792, help="Longitude")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    out_json = "data/weather_cli.json"
    out_csv = "results/weather_cli.csv"

    fetch_weather(args.lat, args.lon, args.city, out_json)
    df = process_weather(out_json, out_csv)
    plot_weather(df, args.city)

if __name__ == "__main__":
    main()
