#!/bin/bash
set -euo pipefail

STREAMLIT_PORT="${STREAMLIT_PORT:-${PORT:-7860}}"
UVICORN_PORT="${UVICORN_PORT:-8000}"
HOST="0.0.0.0"

echo "Environment: FASTAPI_ROOT_PATH=${FASTAPI_ROOT_PATH:-<unset>} PORT=${PORT:-<unset>} STREAMLIT_PORT=${STREAMLIT_PORT:-<unset>} UVICORN_PORT=${UVICORN_PORT}"
echo "Environment: FASTAPI_ROOT_PATH=${FASTAPI_ROOT_PATH:-<unset>} PORT=${PORT:-<unset>} STREAMLIT_PORT=${STREAMLIT_PORT:-<unset>} UVICORN_PORT=${UVICORN_PORT}"
echo "🚀 Starting FastAPI (uvicorn) on port ${UVICORN_PORT}"
python -m uvicorn app.main:app --host "${HOST}" --port "${UVICORN_PORT}" &
UVICORN_PID=$!

cleanup() {
  echo "🛑 Shutting down services..."
  if kill -0 "${UVICORN_PID}" 2>/dev/null; then
    kill "${UVICORN_PID}" 2>/dev/null || true
    wait "${UVICORN_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT="${STREAMLIT_PORT}"
export STREAMLIT_SERVER_ADDRESS="${HOST}"

echo "🌐 Starting Streamlit on port ${STREAMLIT_PORT}"
exec streamlit run streamlit_app.py --server.port "${STREAMLIT_PORT}" --server.address "${HOST}"
