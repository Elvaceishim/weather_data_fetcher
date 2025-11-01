#!/bin/bash
set -euo pipefail
mkdir -p data logs

source .env 2>/dev/null || true
: "${LAT:=6.5244}"
: "${LON:=3.3792}"
: "${CITY:=Lagos}"
: "${PAST_DAYS:=30}"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_FILE=${LOG_FILE:-logs/app.log}

echo "[${STAMP}] Fetching ${PAST_DAYS} past days for ${CITY} (${LAT}, ${LON})"

URL="https://api.open-meteo.com/v1/forecast?latitude=${LAT}&longitude=${LON}&hourly=temperature_2m,relative_humidity_2m,cloudcover,pressure_msl,wind_speed_10m,precipitation,rain&timezone=Africa%2FLagos&past_days=${PAST_DAYS}"

{
  curl -sfL "$URL" -o data/weather.json
  echo "[$STAMP] Saved to data/weather.json"
} | tee -a "$LOG_FILE"
