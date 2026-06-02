"""
Health endpoint: GET /health
Returns service status, last event timestamp per store, STALE_FEED if >10 min lag.
This is what an on-call engineer checks first.
"""

from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models import HealthResponse, StoreHealth
from app.database import EventRecord

STALE_FEED_MINUTES = 10


async def get_health(db: AsyncSession) -> HealthResponse:
    now = datetime.now(timezone.utc)
    as_of = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Get all distinct store IDs and their last event timestamp
    stmt = (
        select(
            EventRecord.store_id,
            func.max(EventRecord.timestamp).label("last_ts")
        )
        .group_by(EventRecord.store_id)
    )
    rows = (await db.execute(stmt)).fetchall()

    store_healths = []
    overall_degraded = False

    for row in rows:
        last_ts = row.last_ts  # naive UTC datetime from DB
        if last_ts is None:
            store_healths.append(StoreHealth(
                store_id=row.store_id,
                status="DOWN",
                last_event_timestamp=None,
                stale_feed=True,
                stale_feed_warning="No events received for this store"
            ))
            overall_degraded = True
            continue

        last_ts_aware = last_ts.replace(tzinfo=timezone.utc)
        lag_minutes = (now - last_ts_aware).total_seconds() / 60
        stale = lag_minutes > STALE_FEED_MINUTES

        if stale:
            overall_degraded = True

        store_healths.append(StoreHealth(
            store_id=row.store_id,
            status="DEGRADED" if stale else "OK",
            last_event_timestamp=last_ts_aware.strftime("%Y-%m-%dT%H:%M:%SZ"),
            stale_feed=stale,
            stale_feed_warning=f"Last event {lag_minutes:.0f}m ago — feed may be stale" if stale else None
        ))

    return HealthResponse(
        status="DEGRADED" if overall_degraded else "OK",
        stores=store_healths,
        checked_at=as_of
    )
