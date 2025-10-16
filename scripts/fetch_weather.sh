#!/bin/bash
set -euo pipefail
mkdir -p data logs
LAT="${1:-6.5244}"   # Lagos
LON="${2:-3.3792}"
URL="https://api.open-meteo.com/v1/forecast?latitude=${LAT}&longitude=${LON}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Africa%2FLagos"
echo "[fetch] lat=$LAT lon=$LON"
curl -sfL "$URL" -o data/weather.json
echo "[fetch] saved: data/weather.json" | tee -a logs/fetch.log
