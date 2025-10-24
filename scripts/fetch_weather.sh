#!/bin/bash
set -euo pipefail
source .env 2>/dev/null || true
mkdir -p data logs

LAT="${LAT:-6.5244}"
LON="${LON:-3.3792}"
CITY="${CITY:-Lagos}"
PAST_DAYS="${PAST_DAYS=60}"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_FILE=${LOG_FILE:-logs/app.log}

URL="https://api.open-meteo.com/v1/forecast?latitude=${LAT}&longitude=${LON}&hourly=temperature_2m,relative_humidity_2m,cloudcover,pressure_msl,wind_speed_10m,precipitation,rain&timezone=Africa%2FLagos&past_days=${PAST_DAYS}"


{
  echo "[$STAMP] Fetching ${PAST_DAYS} past days for ${CITY} (${LAT}, ${LON})"
  curl -sfL "$URL" -o data/weather.json
  echo "[$STAMP] Saved to data/weather.json"
} | tee -a "$LOG_FILE"
