"""
Event ingestion: validate, deduplicate, store.
POST /events/ingest is idempotent by event_id — safe to call twice with same payload.
Partial success: malformed events return invalid status, valid ones proceed.
"""

import logging
from datetime import datetime, timezone
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.models import StoreEvent, IngestRequest, IngestResponse, IngestEventResult
from app.database import EventRecord

log = logging.getLogger("ingestion")


async def ingest_events(request: IngestRequest, db: AsyncSession) -> IngestResponse:
    results: List[IngestEventResult] = []
    accepted = duplicates = invalid = 0

    # Batch-check existing event_ids for deduplication
    event_ids = [e.event_id for e in request.events]
    existing_ids = set()
    if event_ids:
        stmt = select(EventRecord.event_id).where(EventRecord.event_id.in_(event_ids))
        result = await db.execute(stmt)
        existing_ids = {row[0] for row in result.fetchall()}

    seen_in_batch: set[str] = set()

    for event in request.events:
        # Deduplication — idempotent by event_id
        if event.event_id in existing_ids or event.event_id in seen_in_batch:
            results.append(IngestEventResult(
                event_id=event.event_id,
                status="duplicate",
                reason="event_id already ingested"
            ))
            duplicates += 1
            continue

        # Validate timestamp
        try:
            ts = event.ts_datetime()
        except (ValueError, TypeError) as e:
            results.append(IngestEventResult(
                event_id=event.event_id,
                status="invalid",
                reason=f"invalid timestamp: {e}"
            ))
            invalid += 1
            continue

        # Validate confidence range (already done by Pydantic, but belt+suspenders)
        if not (0.0 <= event.confidence <= 1.0):
            results.append(IngestEventResult(
                event_id=event.event_id,
                status="invalid",
                reason="confidence out of range [0,1]"
            ))
            invalid += 1
            continue

        record = EventRecord(
            event_id=event.event_id,
            store_id=event.store_id,
            camera_id=event.camera_id,
            visitor_id=event.visitor_id,
            event_type=event.event_type.value,
            timestamp=ts.replace(tzinfo=None),  # store as UTC naive
            zone_id=event.zone_id,
            dwell_ms=event.dwell_ms,
            is_staff=event.is_staff,
            confidence=event.confidence,
            queue_depth=event.metadata.queue_depth,
            sku_zone=event.metadata.sku_zone,
            session_seq=event.metadata.session_seq
        )
        db.add(record)
        seen_in_batch.add(event.event_id)
        results.append(IngestEventResult(event_id=event.event_id, status="accepted"))
        accepted += 1

    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        log.error(f"DB commit failed: {e}")
        raise

    return IngestResponse(
        accepted=accepted,
        duplicates=duplicates,
        invalid=invalid,
        results=results
    )
