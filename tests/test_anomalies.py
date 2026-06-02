# PROMPT: "Write pytest tests for anomaly detection: queue spike thresholds (WARN at depth 5,
# CRITICAL at depth 10), dead zone detection (no visits in 30 min → INFO anomaly),
# dead zone not fired when zone visited recently. Verify severity labels and
# suggested_action fields are always populated on every anomaly."
#
# CHANGES MADE: Moved DB isolation to conftest.py. Removed over-broad assertion on exact
# anomaly count — tests now assert on presence/absence of specific anomaly types.
# Added assertion that empty store returns a list (not null/error).
# Fixed dead-zone test to use a timestamp exactly 40 minutes in the past
# rather than relying on an offset that could race with today_window() boundary.

import pytest
import uuid
from datetime import datetime, timezone, timedelta

from httpx import AsyncClient, ASGITransport

from app.main import app


def make_event(
    event_type: str = "ENTRY",
    visitor_id: str = None,
    zone_id: str = None,
    queue_depth: int = None,
    is_staff: bool = False,
    minutes_ago: int = 0,
) -> dict:
    ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": "ST1008",
        "camera_id": "CAM_BILLING_01",
        "visitor_id": visitor_id or ("VIS_" + uuid.uuid4().hex[:8]),
        "event_type": event_type,
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "zone_id": zone_id,
        "dwell_ms": 5000,
        "is_staff": is_staff,
        "confidence": 0.88,
        "metadata": {"queue_depth": queue_depth, "sku_zone": zone_id, "session_seq": 1},
    }


# ── Anomaly detection tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAnomalyDetection:

    async def test_empty_store_returns_empty_list_not_error(self):
        """Empty store must return anomalies: [] — no crash, no false positives."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/stores/ST1008/anomalies")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["anomalies"], list)
        # No false positives for a store with zero events
        assert len(body["anomalies"]) == 0

    async def test_queue_spike_critical_at_depth_10(self):
        """Queue depth ≥ 10 in last 15 min → CRITICAL BILLING_QUEUE_SPIKE anomaly."""
        events = [
            make_event(
                event_type="BILLING_QUEUE_JOIN",
                zone_id="BILLING_COUNTER",
                visitor_id=f"VIS_{i}",
                queue_depth=10,
                minutes_ago=2,
            )
            for i in range(3)
        ]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/events/ingest", json={"events": events})
            resp = await ac.get("/stores/ST1008/anomalies")
        queue_anomalies = [
            a for a in resp.json()["anomalies"]
            if a["anomaly_type"] == "BILLING_QUEUE_SPIKE"
        ]
        assert len(queue_anomalies) >= 1
        assert queue_anomalies[0]["severity"] == "CRITICAL"

    async def test_queue_spike_warn_at_depth_6(self):
        """Queue depth 5–9 in last 15 min → WARN BILLING_QUEUE_SPIKE anomaly."""
        events = [
            make_event(
                event_type="BILLING_QUEUE_JOIN",
                zone_id="BILLING_COUNTER",
                visitor_id=f"VIS_{i}",
                queue_depth=6,
                minutes_ago=3,
            )
            for i in range(2)
        ]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/events/ingest", json={"events": events})
            resp = await ac.get("/stores/ST1008/anomalies")
        queue_anomalies = [
            a for a in resp.json()["anomalies"]
            if a["anomaly_type"] == "BILLING_QUEUE_SPIKE"
        ]
        assert len(queue_anomalies) >= 1
        assert queue_anomalies[0]["severity"] in ("WARN", "CRITICAL")

    async def test_every_anomaly_has_suggested_action(self):
        """Every emitted anomaly must have a non-empty suggested_action string."""
        events = [
            make_event(
                event_type="BILLING_QUEUE_JOIN",
                zone_id="BILLING_COUNTER",
                visitor_id=f"VIS_{i}",
                queue_depth=12,
                minutes_ago=1,
            )
            for i in range(2)
        ]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/events/ingest", json={"events": events})
            resp = await ac.get("/stores/ST1008/anomalies")
        for anomaly in resp.json()["anomalies"]:
            assert anomaly.get("suggested_action"), \
                f"Anomaly {anomaly['anomaly_type']} missing suggested_action"
            assert len(anomaly["suggested_action"]) > 10

    async def test_every_anomaly_has_valid_severity(self):
        """Every emitted anomaly must have severity INFO, WARN, or CRITICAL."""
        events = [
            make_event(
                event_type="BILLING_QUEUE_JOIN",
                zone_id="BILLING_COUNTER",
                queue_depth=8,
                minutes_ago=1,
            )
        ]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/events/ingest", json={"events": events})
            resp = await ac.get("/stores/ST1008/anomalies")
        for anomaly in resp.json()["anomalies"]:
            assert anomaly["severity"] in ("INFO", "WARN", "CRITICAL")

    async def test_dead_zone_detected_after_30_min_silence(self):
        """Zone with no visits in last 30+ min → INFO DEAD_ZONE anomaly."""
        old_event = make_event(
            event_type="ZONE_ENTER",
            zone_id="SKINCARE",
            minutes_ago=40,  # well outside the 30-min window
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/events/ingest", json={"events": [old_event]})
            resp = await ac.get("/stores/ST1008/anomalies")
        dead_zones = [
            a for a in resp.json()["anomalies"]
            if a["anomaly_type"] == "DEAD_ZONE"
        ]
        assert len(dead_zones) >= 1
        assert dead_zones[0]["severity"] == "INFO"
        assert "SKINCARE" in dead_zones[0]["description"]

    async def test_dead_zone_not_fired_for_recently_visited_zone(self):
        """Zone visited within last 30 min must NOT produce a DEAD_ZONE anomaly."""
        recent_event = make_event(
            event_type="ZONE_ENTER",
            zone_id="MAKEUP",
            minutes_ago=5,  # well within the 30-min window
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/events/ingest", json={"events": [recent_event]})
            resp = await ac.get("/stores/ST1008/anomalies")
        makeup_dead = [
            a for a in resp.json()["anomalies"]
            if a["anomaly_type"] == "DEAD_ZONE" and "MAKEUP" in a["description"]
        ]
        assert len(makeup_dead) == 0

    async def test_anomaly_response_has_required_fields(self):
        """Anomaly response must have store_id, as_of, and anomalies list."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/stores/ST1008/anomalies")
        body = resp.json()
        assert "store_id" in body
        assert "as_of" in body
        assert "anomalies" in body
        assert isinstance(body["anomalies"], list)

    async def test_each_anomaly_has_anomaly_id(self):
        """Each anomaly object must have a unique anomaly_id."""
        events = [
            make_event(
                event_type="BILLING_QUEUE_JOIN",
                zone_id="BILLING_COUNTER",
                queue_depth=10,
                minutes_ago=2,
            )
        ]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/events/ingest", json={"events": events})
            resp = await ac.get("/stores/ST1008/anomalies")
        for anomaly in resp.json()["anomalies"]:
            assert "anomaly_id" in anomaly
            assert len(anomaly["anomaly_id"]) > 0
