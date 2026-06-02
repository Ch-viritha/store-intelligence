"""
Real-time metric computation for GET /stores/{id}/metrics.
All metrics computed from live DB — never cached from yesterday.
Staff events (is_staff=True) are excluded from all customer metrics.
"""

import logging
from datetime import datetime, timezone, date, timedelta
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct, and_, text

from app.models import StoreMetrics, ZoneDwell
from app.database import EventRecord

log = logging.getLogger("metrics")


def today_window() -> tuple[datetime, datetime]:
    """Return UTC midnight to now for today's metrics."""
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.replace(tzinfo=None), now.replace(tzinfo=None)


async def get_store_metrics(store_id: str, db: AsyncSession) -> StoreMetrics:
    start, end = today_window()

    # ── Unique visitors (customer only, deduplicated by visitor_id) ──────────
    uv_stmt = (
        select(func.count(distinct(EventRecord.visitor_id)))
        .where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "ENTRY",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= start,
            EventRecord.timestamp <= end
        )
    )
    uv_result = await db.execute(uv_stmt)
    unique_visitors = uv_result.scalar() or 0

    # ── Conversion rate via billing zone + POS correlation ───────────────────
    # Visitors who had a BILLING_QUEUE_JOIN or spent time in BILLING_COUNTER
    billing_visitors_stmt = (
        select(func.count(distinct(EventRecord.visitor_id)))
        .where(
            EventRecord.store_id == store_id,
            EventRecord.event_type.in_(["BILLING_QUEUE_JOIN", "ZONE_ENTER"]),
            EventRecord.zone_id == "BILLING_COUNTER",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= start,
            EventRecord.timestamp <= end
        )
    )
    billing_result = await db.execute(billing_visitors_stmt)
    billing_visitors = billing_result.scalar() or 0

    # Abandonment: BILLING_QUEUE_ABANDON events
    abandon_stmt = (
        select(func.count(distinct(EventRecord.visitor_id)))
        .where(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_ABANDON",
            EventRecord.is_staff == False,
            EventRecord.timestamp >= start,
            EventRecord.timestamp <= end
        )
    )
    abandon_result = await db.execute(abandon_stmt)
    abandoned = abandon_result.scalar() or 0

    conversion_rate = round(billing_visitors / max(unique_visitors, 1), 4)
    abandonment_rate = round(abandoned / max(billing_visitors, 1), 4)

    # ── Average dwell per zone ────────────────────────────────────────────────
    dwell_stmt = (
        select(
            EventRecord.zone_id,
            func.avg(EventRecord.dwell_ms).label("avg_dwell"),
            func.count(EventRecord.id).label("visit_count")
        )
        .where(
            EventRecord.store_id == store_id,
            EventRecord.event_type.in_(["ZONE_DWELL", "ZONE_EXIT"]),
            EventRecord.zone_id.isnot(None),
            EventRecord.is_staff == False,
            EventRecord.timestamp >= start,
            EventRecord.timestamp <= end
        )
        .group_by(EventRecord.zone_id)
    )
    dwell_result = await db.execute(dwell_stmt)
    zone_dwells = [
        ZoneDwell(
            zone_id=row.zone_id,
            avg_dwell_ms=round(float(row.avg_dwell or 0), 1),
            visit_count=row.visit_count
        )
        for row in dwell_result.fetchall()
    ]

    # ── Current queue depth ───────────────────────────────────────────────────
    queue_stmt = (
        select(func.max(EventRecord.queue_depth))
        .where(
            EventRecord.store_id == store_id,
            EventRecord.queue_depth.isnot(None),
            EventRecord.timestamp >= start,
            EventRecord.timestamp <= end
        )
    )
    queue_result = await db.execute(queue_stmt)
    current_queue = queue_result.scalar() or 0

    return StoreMetrics(
        store_id=store_id,
        as_of=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        unique_visitors=unique_visitors,
        conversion_rate=conversion_rate,
        avg_dwell_per_zone=zone_dwells,
        current_queue_depth=current_queue,
        abandonment_rate=abandonment_rate
    )
