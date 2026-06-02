"""
Conversion funnel computation: Entry → Zone Visit → Billing Queue → Purchase
Session is the unit. Re-entries must not double-count a visitor.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct

from app.models import StoreFunnel, FunnelStage
from app.database import EventRecord

log = logging.getLogger("funnel")


def today_window():
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.replace(tzinfo=None), now.replace(tzinfo=None)


async def get_store_funnel(store_id: str, db: AsyncSession) -> StoreFunnel:
    start, end = today_window()

    base = [
        EventRecord.store_id == store_id,
        EventRecord.is_staff == False,
        EventRecord.timestamp >= start,
        EventRecord.timestamp <= end
    ]

    # Stage 1: Unique visitors who entered (ENTRY or REENTRY — deduplicated by visitor_id)
    # Re-entries do NOT double-count — we use distinct visitor_id across ENTRY events
    entry_stmt = (
        select(func.count(distinct(EventRecord.visitor_id)))
        .where(*base, EventRecord.event_type.in_(["ENTRY", "REENTRY"]))
    )
    entries = (await db.execute(entry_stmt)).scalar() or 0

    # Stage 2: Visitors who entered at least one product zone
    zone_visitors_stmt = (
        select(func.count(distinct(EventRecord.visitor_id)))
        .where(
            *base,
            EventRecord.event_type.in_(["ZONE_ENTER", "ZONE_DWELL"]),
            EventRecord.zone_id.notin_(["ENTRY_ZONE", "EXIT_ZONE", "BILLING_COUNTER"])
        )
    )
    zone_visitors = (await db.execute(zone_visitors_stmt)).scalar() or 0

    # Stage 3: Visitors who reached billing (including queue join)
    billing_stmt = (
        select(func.count(distinct(EventRecord.visitor_id)))
        .where(
            *base,
            EventRecord.event_type.in_(["BILLING_QUEUE_JOIN", "ZONE_ENTER"]),
            EventRecord.zone_id == "BILLING_COUNTER"
        )
    )
    billing_visitors = (await db.execute(billing_stmt)).scalar() or 0

    # Stage 4: Purchasers — visitors in billing zone who were NOT abandoned
    # Proxy: billing_visitors - abandoned
    abandoned_stmt = (
        select(func.count(distinct(EventRecord.visitor_id)))
        .where(*base, EventRecord.event_type == "BILLING_QUEUE_ABANDON")
    )
    abandoned = (await db.execute(abandoned_stmt)).scalar() or 0
    purchasers = max(0, billing_visitors - abandoned)

    def drop_off(current: int, prev: int) -> float:
        if prev == 0:
            return 0.0
        return round((prev - current) / prev * 100, 1)

    stages = [
        FunnelStage(stage="Entry",        count=entries,         drop_off_pct=0.0),
        FunnelStage(stage="Zone Visit",   count=zone_visitors,   drop_off_pct=drop_off(zone_visitors, entries)),
        FunnelStage(stage="Billing Queue", count=billing_visitors, drop_off_pct=drop_off(billing_visitors, zone_visitors)),
        FunnelStage(stage="Purchase",     count=purchasers,      drop_off_pct=drop_off(purchasers, billing_visitors)),
    ]

    return StoreFunnel(
        store_id=store_id,
        as_of=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        stages=stages
    )
