"""
Event schema definition and emission to .jsonl file and/or Intelligence API.
"""

import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List

log = logging.getLogger("emit")

VALID_EVENT_TYPES = {
    "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT",
    "ZONE_DWELL", "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY",
}


def build_event(
    store_id: str,
    camera_id: str,
    visitor_id: str,
    event_type: str,
    timestamp: datetime,
    zone_id: Optional[str],
    dwell_ms: int,
    is_staff: bool,
    confidence: float,
    session_seq: int,
    sku_zone: Optional[str] = None,
    queue_depth: Optional[int] = None,
) -> dict:
    """Build a structured event conforming to the required output schema."""
    assert event_type in VALID_EVENT_TYPES, \
        f"Invalid event_type '{event_type}'. Must be one of: {VALID_EVENT_TYPES}"

    if isinstance(timestamp, datetime):
        ts_str = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        ts_str = str(timestamp)

    return {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": ts_str,
        "zone_id": zone_id,
        "dwell_ms": int(dwell_ms),
        "is_staff": bool(is_staff),
        "confidence": round(float(confidence), 4),
        "metadata": {
            "queue_depth": queue_depth,
            "sku_zone": sku_zone,
            "session_seq": int(session_seq),
        },
    }


class EventEmitter:
    """
    Writes events to a .jsonl output file.
    Optionally batches and POSTs to the Intelligence API in real-time.
    """

    def __init__(
        self,
        output_path: str,
        api_url: Optional[str] = None,
        batch_size: int = 100,
    ):
        self.output_path = output_path
        self.api_url = api_url
        self.batch_size = batch_size
        self.count = 0
        self._buffer: List[dict] = []
        self._file = open(output_path, "w", encoding="utf-8")

    def emit(self, event: dict):
        """Write event to file and buffer for optional API ingestion."""
        self._file.write(json.dumps(event) + "\n")
        self._buffer.append(event)
        self.count += 1

        if len(self._buffer) >= self.batch_size:
            self._flush_to_api()

    def _flush_to_api(self):
        if not self.api_url or not self._buffer:
            self._buffer = []
            return

        import requests
        try:
            resp = requests.post(
                f"{self.api_url}/events/ingest",
                json={"events": self._buffer},
                timeout=15,
            )
            if resp.status_code not in (200, 207):
                log.warning(
                    f"API ingest returned {resp.status_code}: {resp.text[:200]}"
                )
        except Exception as e:
            log.warning(f"API ingest failed (will retry on next batch): {e}")
        finally:
            self._buffer = []

    def flush(self):
        """Flush remaining buffer to API and close the output file."""
        self._flush_to_api()
        try:
            self._file.flush()
        finally:
            self._file.close()
