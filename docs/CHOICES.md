# CHOICES.md — Key Engineering Decisions

## Decision 1: Detection Model — YOLOv8n + ByteTrack

### Options Considered
| Option | Pros | Cons |
|---|---|---|
| YOLOv8n (chosen) | Fast on CPU, good person detection, native ByteTrack integration | Smaller model — lower accuracy in crowded scenes |
| YOLOv8m | Better accuracy | 3× slower on CPU, impractical for 20-min clips without GPU |
| RT-DETR | Transformer-based, excellent accuracy | Requires GPU for real-time, complex setup |
| MediaPipe | Very fast | Designed for single-person, poor multi-person tracking |

### What AI Suggested
Claude recommended YOLOv8n as the starting point for CPU inference and noted that the
`ultralytics` library includes ByteTrack natively via `model.track(persist=True)`, which
eliminates the need to integrate a separate tracking library. This was accurate — using the
built-in tracking saved significant integration complexity.

Claude also suggested considering RT-DETR for the billing clip (crowded, partial occlusion)
as a secondary model. I evaluated this but decided against running two models for different
clips, as it complicates the pipeline and the problem statement says accuracy is evaluated
against a held-out clip — not the billing clip specifically.

### What I Chose and Why
**YOLOv8n + ByteTrack** via ultralytics. The primary constraint is that this needs to run
on a machine without GPU in a Docker container. YOLOv8n achieves ~20fps on CPU for 1080p
frames; processing every 3rd frame (5fps effective) gives adequate tracking continuity.

The model struggles with partial occlusion (acknowledged in the footage spec). I handle this
by not suppressing low-confidence detections — they are emitted with their actual confidence
score so downstream consumers can apply their own threshold.

---

## Decision 2: Event Schema Design

### Options Considered
1. **Flat schema** — all fields at top level → simple but verbose, harder to evolve
2. **Nested metadata object** (chosen) → groups optional/variable fields, cleaner versioning
3. **Event type-specific schemas** — separate Pydantic model per event_type → type-safe
   but generates 8 different schemas to maintain

### What AI Suggested
Claude suggested option 3 (event-type specific schemas) for maximum type safety, arguing that
BILLING_QUEUE_JOIN having `queue_depth` as a required field is better than an optional nullable.

**I disagreed and chose option 2.** The reason: the ingestion endpoint receives batches of
mixed event types. Having a single schema with a `metadata` object for optional fields means:
1. One validation path instead of 8
2. The detection pipeline emits one format regardless of event type
3. New event types (e.g. `PROMOTIONAL_DISPLAY_DWELL`) can be added without an API change

The trade-off is that `queue_depth` is nullable in the schema — I accept this and validate
it in the anomaly detection logic rather than at ingest time.

### Key Schema Decisions
- `confidence` is always included and never suppressed — flagged, not dropped
- `is_staff` is a first-class field, not inside metadata, because it affects every downstream
  query (every metric excludes staff)
- `session_seq` allows reconstruction of visitor journeys without full event replay
- `visitor_id` is scoped to a visit session, not a person permanently — avoids biometric
  identity concerns while still enabling re-entry detection within a session window

---

## Decision 3: API Architecture — SQLite + FastAPI (async) vs PostgreSQL + Redis

### Options Considered
| Option | Pros | Cons |
|---|---|---|
| SQLite + aiosqlite (chosen) | Zero config, single docker compose, portable | Concurrent write bottleneck at 40 stores |
| PostgreSQL | Production-grade concurrency | Requires second container, more setup |
| PostgreSQL + Redis cache | Best performance for real-time metrics | Significantly more complex for a prototype |

### What AI Suggested
Claude recommended PostgreSQL as the storage engine, noting that SQLite's write locking would
become a bottleneck under concurrent event ingestion from 40 stores simultaneously. It also
suggested Redis for caching hot metrics (conversion_rate, queue_depth) with a 30-second TTL.

**I chose SQLite for this submission and documented why.** The acceptance gate requires
`docker compose up` with no manual steps — adding PostgreSQL requires a separate container
with init scripts and health check dependencies. For a take-home challenge, the operability
tradeoff is worth making.

The code is structured so that switching to PostgreSQL requires only one change: the
`DATABASE_URL` environment variable. The SQLAlchemy async layer abstracts the driver.

### What I Would Change for Production
At 40 live stores with ~500 events per clip per store per 20 minutes:
1. Switch `DATABASE_URL` to PostgreSQL
2. Add Redis for caching `/metrics` and `/heatmap` with 30s TTL
3. Add a background worker for anomaly detection (currently computed on-request)
4. Add TimescaleDB extension on PostgreSQL for time-series query performance

This is documented in DESIGN.md under the "first thing that breaks" section.

---

## VLM Usage — Zone Classification Evaluation

I evaluated using Claude Vision (via the Anthropic API) for zone classification — asking the
model to identify which product zone a person is standing in based on the frame.

**Prompt used:**
> "This is a frame from a retail store CCTV camera. The store sells beauty and personal care
> products. Identify which product zone the highlighted person (bounding box shown in red) is
> standing in. Zones: SKINCARE, MAKEUP, HAIRCARE, PERSONAL_CARE, BATH_BODY, FRAGRANCE,
> BILLING_COUNTER. Reply with just the zone name."

**Result:** Claude Vision was accurate (~85%) for frames where the shelf labels were visible.
It struggled with the overview camera (CAM5) where shelves were not legible, and with the
billing area where the background is a counter rather than product shelving.

**What I chose instead:** Rule-based centroid mapping (bounding box centre mapped to a grid
of zone polygons from store_layout.json). This is faster, fully deterministic, and does not
require an API call per detection (which would be ~900 API calls per minute at 5fps).

The VLM approach would be the right choice for a store where the layout changes frequently
or where camera angles are unusual — document this in production recommendations.
