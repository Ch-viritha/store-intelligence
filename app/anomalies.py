"""
Anomaly detection for GET /stores/{id}/anomalies.
Detects: queue spike, conversion drop vs 7-day avg, dead zone (no visits in 30 min).
Severity: INFO / WARN / CRITICAL with suggested_action per anomaly.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct, and_

from app.models import StoreAnomalies, Anomaly, AnomalySeverity
from app.database import EventRecord

log = logging.getLogger("anomalies")

QUEUE_WARN_DEPTH = 5
QUEUE_CRITICAL_DEPTH = 10
CONVERSION_DROP_WARN_PCT = 20    # 20% drop vs 7-day avg = WARN
CONVERSION_DROP_CRITICAL_PCT = 40
DEAD_ZONE_MINUTES = 30           # no visits in 30 min = INFO anomaly


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


async def _get_conversion_rate(store_id: str, start: datetime, end: datetime, db: AsyncSession) -> float:
    base = [
        EventRecord.store_id == store_id,
        EventRecord.is_staff == False,
        EventRecord.timestamp >= start.replace(tzinfo=None),
        EventRecord.timestamp <= end.replace(tzinfo=None)
    ]
    uv = (await db.execute(
        select(func.count(distinct(EventRecord.visitor_id)))
        .where(*base, EventRecord.event_type.in_(["ENTRY", "REENTRY"]))
    )).scalar() or 0

    bv = (await db.execute(
        select(func.count(distinct(EventRecord.visitor_id)))
        .where(*base, EventRecord.event_type.in_(["BILLING_QUEUE_JOIN", "ZONE_ENTER"]),
               EventRecord.zone_id == "BILLING_COUNTER")
    )).scalar() or 0

    return bv / max(uv, 1)


async def detect_anomalies(store_id: str, db: AsyncSession) -> StoreAnomalies:
    now = now_utc()
    anomalies: List[Anomaly] = []
    as_of = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── 1. Queue spike ────────────────────────────────────────────────────────
    queue_stmt = (
        select(func.max(EventRecord.queue_depth))
        .where(
            EventRecord.store_id == store_id,
            EventRecord.queue_depth.isnot(None),
            EventRecord.timestamp >= (now - timedelta(minutes=15)).replace(tzinfo=None)
        )
    )
    max_queue = (await db.execute(queue_stmt)).scalar() or 0

    if max_queue >= QUEUE_CRITICAL_DEPTH:
        anomalies.append(Anomaly(
            anomaly_type="BILLING_QUEUE_SPIKE",
            severity=AnomalySeverity.CRITICAL,
            description=f"Queue depth reached {max_queue} in last 15 minutes",
            suggested_action="Open additional billing counter immediately. Alert floor manager.",
            detected_at=as_of,
            store_id=store_id
        ))
    elif max_queue >= QUEUE_WARN_DEPTH:
        anomalies.append(Anomaly(
            anomaly_type="BILLING_QUEUE_SPIKE",
            severity=AnomalySeverity.WARN,
            description=f"Queue depth reached {max_queue} in last 15 minutes",
            suggested_action="Monitor queue — consider routing customers to secondary counter.",
            detected_at=as_of,
            store_id=store_id
        ))

    # ── 2. Conversion drop vs 7-day average ───────────────────────────────────
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_rate = await _get_conversion_rate(store_id, today_start, now, db)

    seven_day_rates = []
    for days_ago in range(1, 8):
        d_start = today_start - timedelta(days=days_ago)
        d_end = d_start + timedelta(days=1)
        rate = await _get_conversion_rate(store_id, d_start, d_end, db)
        seven_day_rates.append(rate)

    valid_rates = [r for r in seven_day_rates if r > 0]
    if valid_rates:
        avg_7d = sum(valid_rates) / len(valid_rates)
        if avg_7d > 0:
            drop_pct = (avg_7d - today_rate) / avg_7d * 100
            if drop_pct >= CONVERSION_DROP_CRITICAL_PCT:
                anomalies.append(Anomaly(
                    anomaly_type="CONVERSION_DROP",
                    severity=AnomalySeverity.CRITICAL,
                    description=f"Conversion rate {today_rate:.1%} is {drop_pct:.0f}% below 7-day avg ({avg_7d:.1%})",
                    suggested_action="Check floor staff coverage, pricing displays, and billing counter availability.",
                    detected_at=as_of,
                    store_id=store_id
                ))
            elif drop_pct >= CONVERSION_DROP_WARN_PCT:
                anomalies.append(Anomaly(
                    anomaly_type="CONVERSION_DROP",
                    severity=AnomalySeverity.WARN,
                    description=f"Conversion rate {today_rate:.1%} is {drop_pct:.0f}% below 7-day avg ({avg_7d:.1%})",
                    suggested_action="Review product zone staffing and promotions for today.",
                    detected_at=as_of,
                    store_id=store_id
                ))

    # ── 3. Dead zone — no visits in 30 minutes ────────────────────────────────
    dead_zone_cutoff = (now - timedelta(minutes=DEAD_ZONE_MINUTES)).replace(tzinfo=None)
    active_zones_stmt = (
        select(distinct(EventRecord.zone_id))
        .where(
            EventRecord.store_id == store_id,
            EventRecord.zone_id.isnot(None),
            EventRecord.zone_id.notin_(["ENTRY_ZONE", "EXIT_ZONE"]),
            EventRecord.is_staff == False,
            EventRecord.timestamp >= dead_zone_cutoff
        )
    )
    active_zones = {row[0] for row in (await db.execute(active_zones_stmt)).fetchall()}

    all_zones_stmt = (
        select(distinct(EventRecord.zone_id))
        .where(
            EventRecord.store_id == store_id,
            EventRecord.zone_id.isnot(None),
            EventRecord.zone_id.notin_(["ENTRY_ZONE", "EXIT_ZONE"]),
            EventRecord.is_staff == False
        )
    )
    all_zones = {row[0] for row in (await db.execute(all_zones_stmt)).fetchall()}
    dead_zones = all_zones - active_zones

    for zone in dead_zones:
        anomalies.append(Anomaly(
            anomaly_type="DEAD_ZONE",
            severity=AnomalySeverity.INFO,
            description=f"Zone '{zone}' has had no customer visits in the last {DEAD_ZONE_MINUTES} minutes",
            suggested_action=f"Check if zone '{zone}' display needs restocking or if signage is directing customers away.",
            detected_at=as_of,
            store_id=store_id
        ))

    return StoreAnomalies(store_id=store_id, as_of=as_of, anomalies=anomalies)
