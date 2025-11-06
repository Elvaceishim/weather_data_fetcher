#!/bin/bash
set -euo pipefail

STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
UVICORN_PORT="${UVICORN_PORT:-${PORT:-8000}}"
HOST="0.0.0.0"

echo "Environment: PORT=${PORT:-<unset>} STREAMLIT_PORT=${STREAMLIT_PORT} UVICORN_PORT=${UVICORN_PORT}"

export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT="${STREAMLIT_PORT}"
export STREAMLIT_SERVER_ADDRESS="${HOST}"

echo "🌐 Starting Streamlit on port ${STREAMLIT_PORT}"
streamlit run streamlit_app.py --server.port "${STREAMLIT_PORT}" --server.address "${HOST}" &
STREAMLIT_PID=$!

cleanup() {
  echo "🛑 Shutting down services..."
  if kill -0 "${STREAMLIT_PID}" 2>/dev/null; then
    kill "${STREAMLIT_PID}" 2>/dev/null || true
    wait "${STREAMLIT_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "🚀 Starting FastAPI (uvicorn) on port ${UVICORN_PORT}"
exec python -m uvicorn app.main:app --host "${HOST}" --port "${UVICORN_PORT}" --proxy-headers
