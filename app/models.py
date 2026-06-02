"""
Pydantic models for event schema, API requests and responses.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, List
from datetime import datetime
from enum import Enum
import uuid


class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: int = 0


class StoreEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: EventType
    timestamp: str  # ISO-8601 UTC string
    zone_id: Optional[str] = None
    dwell_ms: int = 0
    is_staff: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("confidence")
    @classmethod
    def confidence_precision(cls, v):
        return round(v, 4)

    def ts_datetime(self) -> datetime:
        return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))


class IngestRequest(BaseModel):
    events: List[StoreEvent] = Field(max_length=500, min_length=1)


class IngestEventResult(BaseModel):
    event_id: str
    status: Literal["accepted", "duplicate", "invalid"]
    reason: Optional[str] = None


class IngestResponse(BaseModel):
    accepted: int
    duplicates: int
    invalid: int
    results: List[IngestEventResult]


class ZoneDwell(BaseModel):
    zone_id: str
    avg_dwell_ms: float
    visit_count: int


class StoreMetrics(BaseModel):
    store_id: str
    as_of: str
    unique_visitors: int
    conversion_rate: float
    avg_dwell_per_zone: List[ZoneDwell]
    current_queue_depth: int
    abandonment_rate: float


class FunnelStage(BaseModel):
    stage: str
    count: int
    drop_off_pct: float


class StoreFunnel(BaseModel):
    store_id: str
    as_of: str
    stages: List[FunnelStage]


class HeatmapZone(BaseModel):
    zone_id: str
    visit_frequency: float   # normalised 0-100
    avg_dwell_ms: float
    data_confidence: bool    # False if <20 sessions in window


class StoreHeatmap(BaseModel):
    store_id: str
    as_of: str
    zones: List[HeatmapZone]


class AnomalySeverity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class Anomaly(BaseModel):
    anomaly_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    anomaly_type: str
    severity: AnomalySeverity
    description: str
    suggested_action: str
    detected_at: str
    store_id: str


class StoreAnomalies(BaseModel):
    store_id: str
    as_of: str
    anomalies: List[Anomaly]


class StoreHealth(BaseModel):
    store_id: str
    status: Literal["OK", "DEGRADED", "DOWN"]
    last_event_timestamp: Optional[str]
    stale_feed: bool
    stale_feed_warning: Optional[str] = None


class HealthResponse(BaseModel):
    service: str = "store-intelligence-api"
    status: Literal["OK", "DEGRADED"]
    stores: List[StoreHealth]
    checked_at: str
