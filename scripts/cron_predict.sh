#!/usr/bin/env bash

set -euo pipefail

# --- Resolve repo root ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- Lock to avoid overlapping runs (portable; no flock needed) ---
mkdir -p logs
LOCKDIR="logs/.predict.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "[$(date '+%F %T')] Another run is in progress. Skipping."
  exit 0
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

# --- Args & defaults ---
MODE="${1:-default}"
CITY="${2:-Lagos}"
LAT="${3:-6.5244}"
LON="${4:-3.3792}"
PAST_DAYS="${5:-90}"

# --- Activate venv if present ---
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# --- Environment for fetch scripts ---
export LAT="$LAT" LON="$LON" PAST_DAYS="$PAST_DAYS"

# --- Run one logged prediction ---
echo "[$(date '+%F %T')] cron_predict: city=$CITY lat=$LAT lon=$LON mode=$MODE days=$PAST_DAYS"
python3 scripts/log_predict.py --city "$CITY" --lat "$LAT" --lon "$LON" --mode "$MODE" || {
  echo "[$(date '+%F %T')] ERROR: log_predict failed"
  exit 1
}

# --- (Optional) basic log rotation (keep log under ~1MB) ---
LOGFILE="logs/cron.log"
if [[ -f "$LOGFILE" ]] && [[ $(stat -f%z "$LOGFILE") -gt 1048576 ]]; then
  mv "$LOGFILE" "logs/cron_$(date +%Y%m%d_%H%M%S).log" || true
fi

echo "[$(date '+%F %T')] cron_predict: done."
