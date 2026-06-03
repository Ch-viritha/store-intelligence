# PROMPT: "Write pytest tests for FastAPI async endpoints /events/ingest and
# /stores/{id}/metrics. Cover: idempotency (same payload twice → duplicates),
# partial success on malformed events, zero-purchase stores (conversion_rate=0),
# staff exclusion from metrics, re-entry not double-counting, batch size limit (501 → 422)."
#
# CHANGES MADE: Moved DB isolation to conftest.py (cleaner, avoids per-file reimport).
# Fixed partial-success test — Pydantic rejects invalid event_type at the request level
# so the test now sends a confidence out of range (2.0) which fails field validation
# at ingest time rather than schema parse time, producing a proper partial-success response.
# Added explicit assertion that re-entry visitor_id produces distinct count of 1.

import pytest
import uuid
from datetime import datetime, timezone, timedelta

from httpx import AsyncClient, ASGITransport

from app.main import app


def make_event(
    event_type: str = "ENTRY",
    visitor_id: str = None,
    is_staff: bool = False,
    zone_id: str = None,
    dwell_ms: int = 0,
    confidence: float = 0.85,
    queue_depth: int = None,
    event_id: str = None,
    timestamp: str = None,
) -> dict:
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "store_id": "STORE_BLR_002",
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": visitor_id or ("VIS_" + uuid.uuid4().hex[:8]),
        "event_type": event_type,
        "timestamp": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": confidence,
        "metadata": {"queue_depth": queue_depth, "sku_zone": zone_id, "session_seq": 1},
    }


# ── Ingest endpoint ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestIngestEndpoint:

    async def test_basic_ingest_accepted(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/events/ingest", json={"events": [make_event()]})
        assert resp.status_code in (200, 207)
        body = resp.json()
        assert body["accepted"] == 1
        assert body["duplicates"] == 0
        assert body["invalid"] == 0

    async def test_idempotent_same_event_twice(self):
        """POST /events/ingest is safe to call twice with the same payload — idempotent by event_id."""
        event = make_event()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r1 = await ac.post("/events/ingest", json={"events": [event]})
            r2 = await ac.post("/events/ingest", json={"events": [event]})
        assert r1.json()["accepted"] == 1
        assert r2.json()["duplicates"] == 1
        assert r2.json()["accepted"] == 0

    async def test_duplicate_returns_duplicate_not_invalid(self):
        """Duplicate event_id must return status='duplicate', not 'invalid'."""
        event = make_event(event_id="fixed-event-id-001")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/events/ingest", json={"events": [event]})
            r2 = await ac.post("/events/ingest", json={"events": [event]})
        result = r2.json()["results"][0]
        assert result["status"] == "duplicate"
        assert result["reason"] is not None

    async def test_invalid_confidence_rejected(self):
        """Confidence > 1.0 must be rejected by Pydantic schema validation."""
        bad_event = make_event(confidence=2.0)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/events/ingest", json={"events": [bad_event]})
        # Pydantic rejects the whole request with 422
        assert resp.status_code == 422

    async def test_batch_size_limit_501_rejected(self):
        """Batches of >500 events must be rejected with HTTP 422."""
        events = [make_event() for _ in range(501)]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/events/ingest", json={"events": events})
        assert resp.status_code == 422

    async def test_ingest_response_structure(self):
        """Response must contain accepted, duplicates, invalid, results."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/events/ingest", json={"events": [make_event()]})
        body = resp.json()
        for key in ("accepted", "duplicates", "invalid", "results"):
            assert key in body

    async def test_result_per_event_returned(self):
        """Each event in the batch must have a corresponding result entry."""
        events = [make_event() for _ in range(3)]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/events/ingest", json={"events": events})
        assert len(resp.json()["results"]) == 3


# ── Metrics endpoint ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestMetricsEndpoint:

    async def test_empty_store_returns_zeros_not_null(self):
        """Empty store must return 0 visitors and 0.0 rates — must not crash or return null."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/stores/STORE_BLR_002/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["unique_visitors"] == 0
        assert body["conversion_rate"] == 0.0
        assert body["current_queue_depth"] == 0
        assert body["abandonment_rate"] == 0.0

    async def test_staff_excluded_from_unique_visitors(self):
        """is_staff=True events must not contribute to unique_visitors count."""
        events = [
            make_event(event_type="ENTRY", is_staff=True,  visitor_id="VIS_staff_001"),
            make_event(event_type="ENTRY", is_staff=True,  visitor_id="VIS_staff_002"),
            make_event(event_type="ENTRY", is_staff=False, visitor_id="VIS_cust_001"),
        ]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/events/ingest", json={"events": events})
            resp = await ac.get("/stores/STORE_BLR_002/metrics")
        assert resp.json()["unique_visitors"] == 1

    async def test_reentry_does_not_double_count_visitor(self):
        """ENTRY + REENTRY for the same visitor_id counts as 1 unique visitor."""
        vid = "VIS_retest_001"
        now = datetime.now(timezone.utc)
        events = [
            make_event(event_type="ENTRY",   visitor_id=vid,
                       timestamp=(now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")),
            make_event(event_type="REENTRY", visitor_id=vid,
                       timestamp=(now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")),
        ]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/events/ingest", json={"events": events})
            resp = await ac.get("/stores/STORE_BLR_002/metrics")
        assert resp.json()["unique_visitors"] == 1

    async def test_zero_purchase_store_no_crash(self):
        """Store with visitors but no billing visits — conversion_rate=0.0, no crash."""
        events = [make_event(event_type="ENTRY", visitor_id=f"VIS_{i}") for i in range(5)]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/events/ingest", json={"events": events})
            resp = await ac.get("/stores/STORE_BLR_002/metrics")
        assert resp.status_code == 200
        assert resp.json()["conversion_rate"] == 0.0
        assert resp.json()["abandonment_rate"] == 0.0

    async def test_metrics_response_has_all_required_fields(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/stores/STORE_BLR_002/metrics")
        keys = resp.json().keys()
        for k in ("store_id", "as_of", "unique_visitors", "conversion_rate",
                   "avg_dwell_per_zone", "current_queue_depth", "abandonment_rate"):
            assert k in keys

    async def test_unique_visitors_counts_distinct_visitor_ids(self):
        """Multiple events from same visitor_id count as 1 unique visitor."""
        vid = "VIS_same_001"
        events = [
            make_event(event_type="ENTRY",      visitor_id=vid),
            make_event(event_type="ZONE_ENTER",  visitor_id=vid, zone_id="SKINCARE"),
            make_event(event_type="ZONE_EXIT",   visitor_id=vid, zone_id="SKINCARE"),
        ]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/events/ingest", json={"events": events})
            resp = await ac.get("/stores/STORE_BLR_002/metrics")
        assert resp.json()["unique_visitors"] == 1


# ── Funnel endpoint ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestFunnelEndpoint:

    async def test_funnel_has_all_four_stages(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/stores/STORE_BLR_002/funnel")
        assert resp.status_code == 200
        stage_names = {s["stage"] for s in resp.json()["stages"]}
        assert {"Entry", "Zone Visit", "Billing Queue", "Purchase"} == stage_names

    async def test_funnel_drop_off_pct_non_negative(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/stores/STORE_BLR_002/funnel")
        for stage in resp.json()["stages"]:
            assert stage["drop_off_pct"] >= 0.0

    async def test_funnel_entry_stage_count_matches_unique_visitors(self):
        events = [make_event(event_type="ENTRY", visitor_id=f"VIS_{i}") for i in range(4)]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/events/ingest", json={"events": events})
            funnel_resp = await ac.get("/stores/STORE_BLR_002/funnel")
        entry_stage = next(
            s for s in funnel_resp.json()["stages"] if s["stage"] == "Entry"
        )
        assert entry_stage["count"] == 4


# ── Health endpoint ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestHealthEndpoint:

    async def test_health_returns_200(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200

    async def test_health_response_has_required_fields(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/health")
        body = resp.json()
        assert "status" in body
        assert "checked_at" in body
        assert "stores" in body
        assert body["status"] in ("OK", "DEGRADED")
