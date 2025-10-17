#!/bin/bash
set -euo pipefail
source .env 2>/dev/null || true
mkdir -p data logs

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_FILE=${LOG_FILE:-logs/app.log}

{
  echo "[$STAMP] Fetching weather for ${CITY:-Lagos} (${LAT:-6.5244}, ${LON:-3.3792})"
  URL="https://api.open-meteo.com/v1/forecast?latitude=${LAT:-6.5244}&longitude=${LON:-3.3792}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Africa%2FLagos"
  curl -sfL "$URL" -o data/weather.json
  echo "[$STAMP] Saved to data/weather.json"
} | tee -a "$LOG_FILE"
