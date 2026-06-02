"""
Zone heatmap: visit frequency + avg dwell, normalised 0-100.
Includes data_confidence flag if fewer than 20 sessions in window.
"""

from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct

from app.models import StoreHeatmap, HeatmapZone
from app.database import EventRecord

MIN_SESSIONS_FOR_CONFIDENCE = 20


def today_window():
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.replace(tzinfo=None), now.replace(tzinfo=None)


async def get_store_heatmap(store_id: str, db: AsyncSession) -> StoreHeatmap:
    start, end = today_window()

    stmt = (
        select(
            EventRecord.zone_id,
            func.count(EventRecord.id).label("visit_count"),
            func.avg(EventRecord.dwell_ms).label("avg_dwell"),
            func.count(distinct(EventRecord.visitor_id)).label("unique_visitors")
        )
        .where(
            EventRecord.store_id == store_id,
            EventRecord.zone_id.isnot(None),
            EventRecord.zone_id.notin_(["ENTRY_ZONE", "EXIT_ZONE"]),
            EventRecord.event_type.in_(["ZONE_ENTER", "ZONE_DWELL", "ZONE_EXIT"]),
            EventRecord.is_staff == False,
            EventRecord.timestamp >= start,
            EventRecord.timestamp <= end
        )
        .group_by(EventRecord.zone_id)
    )
    rows = (await db.execute(stmt)).fetchall()

    if not rows:
        return StoreHeatmap(
            store_id=store_id,
            as_of=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            zones=[]
        )

    # Normalise visit_count to 0-100
    max_visits = max(r.visit_count for r in rows)

    zones = []
    for row in rows:
        freq_normalised = round((row.visit_count / max(max_visits, 1)) * 100, 1)
        has_confidence = row.unique_visitors >= MIN_SESSIONS_FOR_CONFIDENCE
        zones.append(HeatmapZone(
            zone_id=row.zone_id,
            visit_frequency=freq_normalised,
            avg_dwell_ms=round(float(row.avg_dwell or 0), 1),
            data_confidence=has_confidence
        ))

    # Sort by frequency descending
    zones.sort(key=lambda z: z.visit_frequency, reverse=True)

    return StoreHeatmap(
        store_id=store_id,
        as_of=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        zones=zones
    )
