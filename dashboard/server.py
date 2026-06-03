"""
Live Dashboard server — WebSocket-powered real-time store metrics.
Polls the Intelligence API every 5 seconds and pushes updates to all connected browsers.
Part E: proves the pipeline and API are genuinely connected end-to-end.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

log = logging.getLogger("dashboard")

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")
STORE_ID = os.getenv("STORE_ID", "STORE_BLR_002")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SEC", "5"))

STATIC_DIR = Path(__file__).parent / "static"


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        log.info(f"Dashboard client connected ({len(self.active)} total)")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


async def fetch_dashboard_data() -> dict:
    """Fetch all dashboard data from the Intelligence API concurrently."""
    async with httpx.AsyncClient(timeout=8.0) as client:
        endpoints = {
            "metrics":   f"{API_BASE}/stores/{STORE_ID}/metrics",
            "funnel":    f"{API_BASE}/stores/{STORE_ID}/funnel",
            "heatmap":   f"{API_BASE}/stores/{STORE_ID}/heatmap",
            "anomalies": f"{API_BASE}/stores/{STORE_ID}/anomalies",
            "health":    f"{API_BASE}/health",
        }
        results = {}
        for key, url in endpoints.items():
            try:
                r = await client.get(url)
                results[key] = r.json() if r.status_code == 200 else {"error": r.status_code}
            except Exception as e:
                results[key] = {"error": str(e)}
        return results


async def _broadcast_loop():
    """Background task: poll Intelligence API and push updates to WebSocket clients."""
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        if manager.active:
            try:
                data = await fetch_dashboard_data()
                await manager.broadcast({"type": "update", "data": data})
            except Exception as e:
                log.error(f"Broadcast loop error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_broadcast_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Store Intelligence Dashboard", lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send an immediate snapshot on connect so the UI isn't blank
        data = await fetch_dashboard_data()
        await ws.send_json({"type": "snapshot", "data": data})
        # Keep the connection alive — client sends pings
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as e:
        log.warning(f"WebSocket error: {e}")
        manager.disconnect(ws)


@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>Dashboard static files not found</h1>", status_code=500)
    return HTMLResponse(index.read_text(encoding="utf-8"))
