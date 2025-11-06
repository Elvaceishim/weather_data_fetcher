
import asyncio
from pathlib import Path
from datetime import datetime
import os, json, subprocess

import joblib
import numpy as np
import pandas as pd
import httpx
from httpx import RequestError
from fastapi import FastAPI, Query, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field
import websockets

# --------- Paths ---------
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SCRIPTS = ROOT / "scripts"

MODEL_PATH = MODELS / "rain_xgb_tuned.joblib"
META_PATH  = MODELS / "rain_xgb_tuned_meta.json"
HOURLY_CSV = RESULTS / "hourly.csv"

STREAMLIT_PORT = os.getenv("STREAMLIT_PORT", "8501")
STREAMLIT_BASE = os.getenv("STREAMLIT_BASE", f"http://127.0.0.1:{STREAMLIT_PORT}")

import sys
sys.path.insert(0, str(ROOT))
from scripts.train_xgb_tuned_final import build_features  # reuse your exact feature builder

# --------- Load model + meta at startup ---------
model = joblib.load(MODEL_PATH)
meta = json.loads(META_PATH.read_text())
FEATURES = meta["features"]
THRESH = meta["thresholds"]
HORIZON_H = int(meta["horizon_hours"])

app = FastAPI(title="Rain Nowcast API", version="1.0.0")

# --------- Helpers ---------
def ensure_hourly(lat: float, lon: float, past_days: int = 90) -> pd.DataFrame:
    env = os.environ.copy()
    env["LAT"] = str(lat); env["LON"] = str(lon); env["PAST_DAYS"] = str(past_days)
    need_refresh = True
    if HOURLY_CSV.exists():
        age_hours = (datetime.now().timestamp() - HOURLY_CSV.stat().st_mtime) / 3600
        need_refresh = age_hours > 6
    if (not HOURLY_CSV.exists()) or need_refresh:
        subprocess.run(["bash", str(SCRIPTS / "fetch_weather.sh")], check=True, env=env)
        subprocess.run(["python3", str(SCRIPTS / "export_hourly.py")], check=True, env=env)
    return pd.read_csv(HOURLY_CSV, parse_dates=["time"])

def predict_latest(df: pd.DataFrame, mode: str):
    Xdf = build_features(df.copy())
    Xdf = Xdf[FEATURES]
    x = Xdf.iloc[[-1]].values
    p = float(model.predict_proba(x)[0, 1])
    thr_map = {
        "default":   float(THRESH["default"]),
        "recall":    float(THRESH["high_recall"]),
        "precision": float(THRESH["high_precision"]),
    }
    t = thr_map[mode]
    decision = "RAIN" if p >= t else "No rain"
    ts = df.loc[Xdf.index, "time"].iloc[-1]
    return dict(
        timestamp=str(ts),
        probability=p,
        threshold=t,
        mode=mode,
        decision=decision,
        horizon_hours=HORIZON_H,
    )

# --------- Schemas ---------
class PredictBody(BaseModel):
    lat: float = Field(6.5244, description="Latitude")
    lon: float = Field(3.3792, description="Longitude")
    mode: str = Field("default", description="default | recall | precision")
    past_days: int = Field(90, ge=14, le=180, description="How much history to fetch (days)")

# --------- Endpoints ---------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_file": MODEL_PATH.name,
        "horizon_hours": HORIZON_H,
        "thresholds": THRESH,
        "features": FEATURES,
    }

@app.post("/predict")
def predict(body: PredictBody):
    df = ensure_hourly(body.lat, body.lon, body.past_days)
    out = predict_latest(df, body.mode)
    return {"ok": True, "result": out}

@app.get("/predict")
def predict_get(
    lat: float = Query(6.5244), lon: float = Query(3.3792),
    mode: str = Query("default"), past_days: int = Query(90)
):
    df = ensure_hourly(lat, lon, past_days)
    out = predict_latest(df, mode)
    return {"ok": True, "result": out}


def _target_authority(parsed: httpx.URL) -> str:
    if parsed.port and not (
        (parsed.scheme == "http" and parsed.port == 80) or
        (parsed.scheme == "https" and parsed.port == 443)
    ):
        return f"{parsed.host}:{parsed.port}"
    return parsed.host


@app.websocket("/_stcore/stream")
async def proxy_streamlit_ws(websocket: WebSocket):
    base = httpx.URL(STREAMLIT_BASE)
    ws_scheme = "wss" if base.scheme == "https" else "ws"
    authority = _target_authority(base)
    target_url = f"{ws_scheme}://{authority}{websocket.scope['path']}"

    raw_query = websocket.scope.get("query_string", b"")
    if raw_query:
        target_url = f"{target_url}?{raw_query.decode()}"

    extra_headers = []
    for key, value in websocket.headers.items():
        lk = key.lower()
        if lk in {"host", "origin", "sec-websocket-protocol"}:
            continue
        extra_headers.append((key, value))
    extra_headers.append(("Host", authority))
    extra_headers.append(("Origin", f"{base.scheme}://{authority}"))

    sub_header = websocket.headers.get("sec-websocket-protocol")
    client_subprotocols = [proto.strip() for proto in sub_header.split(",")] if sub_header else []
    preferred_subprotocol = client_subprotocols[0] if client_subprotocols else None

    try:
        async with websockets.connect(
            target_url,
            extra_headers=extra_headers,
            subprotocols=client_subprotocols or None,
        ) as upstream:
            await websocket.accept(subprotocol=upstream.subprotocol or preferred_subprotocol)

            async def client_to_upstream():
                try:
                    while True:
                        message = await websocket.receive()
                        if message.get("type") == "websocket.disconnect":
                            await upstream.close()
                            break
                        text = message.get("text")
                        if text is not None:
                            await upstream.send(text)
                        else:
                            data = message.get("bytes")
                            if data is not None:
                                await upstream.send(data)
                except WebSocketDisconnect:
                    await upstream.close()

            async def upstream_to_client():
                try:
                    async for payload in upstream:
                        if isinstance(payload, bytes):
                            await websocket.send_bytes(payload)
                        else:
                            await websocket.send_text(payload)
                finally:
                    await websocket.close()

            tasks = [
                asyncio.create_task(client_to_upstream()),
                asyncio.create_task(upstream_to_client()),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
    except Exception:
        await websocket.close(code=1011)


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"], include_in_schema=False)
async def proxy_streamlit(full_path: str, request: Request):
    """Proxy remaining requests over to the colocated Streamlit server."""
    # Preserve the incoming path while defaulting to root.
    relative_path = f"/{full_path}" if full_path else "/"
    target = httpx.URL(STREAMLIT_BASE).join(relative_path)
    target_url = str(target)
    query_items = tuple(request.query_params.multi_items())

    headers = dict(request.headers)
    target_authority = _target_authority(target)
    headers["host"] = target_authority
    if "origin" in headers:
        headers["origin"] = f"{target.scheme}://{target_authority}"

    body = await request.body()

    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            proxied_response = await client.request(
                request.method,
                target_url,
                params=query_items if query_items else None,
                content=body if body else None,
                headers=headers,
                cookies=request.cookies,
                timeout=30.0,
            )
        except RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Streamlit backend unavailable: {exc}") from exc

    blocked_headers = {"content-encoding", "transfer-encoding", "connection", "content-length"}
    response_headers = {k: v for k, v in proxied_response.headers.items() if k.lower() not in blocked_headers}

    return Response(
        content=proxied_response.content,
        status_code=proxied_response.status_code,
        headers=response_headers,
        media_type=proxied_response.headers.get("content-type"),
    )
