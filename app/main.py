"""
FastAPI entrypoint — Store Intelligence API.
All endpoints are production-aware with structured logging, trace IDs, and graceful degradation.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import OperationalError

from app.database import init_db, get_db
from app.models import IngestRequest, IngestResponse, StoreMetrics, StoreFunnel, StoreHeatmap, StoreAnomalies, HealthResponse
from app.ingestion import ingest_events
from app.metrics import get_store_metrics
from app.funnel import get_store_funnel
from app.heatmap import get_store_heatmap
from app.anomalies import detect_anomalies
from app.health import get_health

# ── Structured logging setup ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
)
log = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    log.info('{"event":"startup","message":"Database initialised"}')
    yield
    log.info('{"event":"shutdown","message":"API shutting down"}')


app = FastAPI(
    title="Store Intelligence API",
    description="AI-powered retail store analytics from CCTV footage",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as e:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        log.error(
            f'{{"trace_id":"{trace_id}","endpoint":"{request.url.path}",'
            f'"latency_ms":{latency_ms},"status_code":500,"error":"{str(e)}"}}'
        )
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "trace_id": trace_id}
        )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    store_id = request.path_params.get("store_id", "")
    log.info(
        f'{{"trace_id":"{trace_id}","store_id":"{store_id}",'
        f'"endpoint":"{request.url.path}","method":"{request.method}",'
        f'"latency_ms":{latency_ms},"status_code":{response.status_code}}}'
    )
    response.headers["X-Trace-Id"] = trace_id
    return response


# ── DB availability guard ─────────────────────────────────────────────────────
async def get_db_or_503(db: AsyncSession = Depends(get_db)):
    try:
        yield db
    except OperationalError:
        raise HTTPException(
            status_code=503,
            detail={"error": "database_unavailable", "message": "Service temporarily unavailable"}
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/events/ingest",
    response_model=IngestResponse,
    summary="Ingest a batch of store events",
    responses={
        207: {"description": "Partial success — some events invalid or duplicate"},
        503: {"description": "Database unavailable"}
    }
)
async def ingest(request: IngestRequest, db: AsyncSession = Depends(get_db_or_503)):
    """
    Accepts batches of up to 500 events. Validates, deduplicates, stores.
    Idempotent by event_id. Returns partial success on malformed events.
    """
    result = await ingest_events(request, db)
    status_code = 207 if (result.invalid > 0 or result.duplicates > 0) else 200
    return Response(
        content=result.model_dump_json(),
        status_code=status_code,
        media_type="application/json"
    )


@app.get(
    "/stores/{store_id}/metrics",
    response_model=StoreMetrics,
    summary="Real-time store metrics"
)
async def metrics(store_id: str, db: AsyncSession = Depends(get_db_or_503)):
    """
    Today's metrics: unique visitors, conversion rate, avg dwell per zone,
    queue depth, abandonment rate. Excludes is_staff=true. Real-time — not cached.
    """
    return await get_store_metrics(store_id, db)


@app.get(
    "/stores/{store_id}/funnel",
    response_model=StoreFunnel,
    summary="Conversion funnel"
)
async def funnel(store_id: str, db: AsyncSession = Depends(get_db_or_503)):
    """
    Conversion funnel: Entry → Zone Visit → Billing Queue → Purchase.
    Session is the unit. Re-entries do not double-count visitors.
    """
    return await get_store_funnel(store_id, db)


@app.get(
    "/stores/{store_id}/heatmap",
    response_model=StoreHeatmap,
    summary="Zone visit heatmap"
)
async def heatmap(store_id: str, db: AsyncSession = Depends(get_db_or_503)):
    """
    Zone visit frequency + avg dwell, normalised 0-100.
    Includes data_confidence=false if fewer than 20 sessions in window.
    """
    return await get_store_heatmap(store_id, db)


@app.get(
    "/stores/{store_id}/anomalies",
    response_model=StoreAnomalies,
    summary="Active anomaly detection"
)
async def anomalies(store_id: str, db: AsyncSession = Depends(get_db_or_503)):
    """
    Detects: queue spike, conversion drop vs 7-day avg, dead zones.
    Severity: INFO / WARN / CRITICAL with suggested_action per anomaly.
    """
    return await detect_anomalies(store_id, db)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check"
)
async def health(db: AsyncSession = Depends(get_db_or_503)):
    """
    Service status, last event timestamp per store.
    Returns STALE_FEED warning if any store feed is >10 min old.
    """
    return await get_health(db)
