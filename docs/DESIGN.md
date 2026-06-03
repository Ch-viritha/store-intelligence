# DESIGN.md — Store Intelligence System

## Architecture Overview

This system converts raw CCTV footage from Purplle's Brigade Road, Bangalore store (STORE_BLR_002)
into a live analytics API. The pipeline has four stages that connect end-to-end:

```
Raw CCTV (CAM1–CAM5)
       │
       ▼
┌─────────────────────────────────────────┐
│  Detection Layer  (pipeline/detect.py) │
│  YOLOv8n → ByteTrack → Re-ID (HSV)    │
│  Staff classifier (uniform hue)        │
│  Zone classifier (centroid mapping)    │
└────────────────┬────────────────────────┘
                 │ structured events (.jsonl)
                 ▼
┌─────────────────────────────────────────┐
│  Intelligence API  (app/main.py)       │
│  FastAPI + SQLite (aiosqlite)          │
│  POST /events/ingest  (idempotent)     │
│  GET  /stores/{id}/metrics             │
│  GET  /stores/{id}/funnel              │
│  GET  /stores/{id}/heatmap             │
│  GET  /stores/{id}/anomalies           │
│  GET  /health                          │
└────────────────┬────────────────────────┘
                 │ WebSocket push (5s poll)
                 ▼
┌─────────────────────────────────────────┐
│  Live Dashboard  (dashboard/server.py) │
│  FastAPI + WebSocket + HTML/JS         │
│  KPIs · Funnel · Heatmap · Anomalies   │
└─────────────────────────────────────────┘
```

## Component Decisions

### Detection Layer

**YOLOv8n** was chosen over larger YOLO variants and RT-DETR for its speed on CPU-only
infrastructure (which most retail deployments run). Processing at 5fps effective (every 3rd
frame at 15fps source) gives adequate tracking while keeping latency under 2 minutes per
20-minute clip on a modern laptop.

**ByteTrack** (via ultralytics `track()` method) was preferred over DeepSORT because it does
not require a separate Re-ID model for tracking continuity — it uses IoU matching as primary
signal and appearance as fallback. This makes it significantly faster while maintaining good
track persistence across occlusions.

**Re-ID approach**: HSV colour histogram embeddings (96-dimensional: 32 bins each for H, S, V)
compared via cosine similarity. This is a deliberate trade-off:
- OSNet/torchreid would give better Re-ID accuracy (especially for re-entry after long gaps)
  but requires GPU or significantly longer processing time
- The histogram approach works well for the primary retail use case: deduplicating the same
  person across overlapping camera angles within a short time window (2 seconds)
- For re-entry after 5–10 minutes, the similarity threshold (0.82) is conservative enough
  to avoid false matches between different customers

**Staff detection**: Purplle store staff wear a distinctive uniform. We detect this via
dominant green/teal hue in the torso region of the bounding box using HSV colour masking.
This avoids the need for a fine-tuned uniform classifier model while handling the primary
case. Limitations: customers wearing similar colours could be misclassified — confidence
score is set accordingly and flagged rather than suppressed.

### Event Schema Design

The schema includes `confidence` (not suppressed for low values — only flagged) because
production CV systems are never perfect. Suppressing low-confidence events would silently
lose real detections; flagging them preserves data for downstream quality analysis.

`is_staff` is included in every event (not just flagged events) because staff exclusion is
applied at the API layer, not the detection layer. This allows the API to audit staff movement
separately if needed.

`session_seq` provides an ordinal counter per visitor session, which enables downstream
detection of session gaps and re-entry patterns without requiring full event replay.

### Intelligence API

**FastAPI** with async SQLite (aiosqlite) was chosen for:
1. Single `docker compose up` deployment with no external services
2. Fast async I/O for concurrent event ingestion during pipeline replay
3. Pydantic v2 for automatic schema validation with clear error messages

All metric queries compute from live data. There is no caching layer — the problem statement
explicitly requires real-time, not cached from yesterday. At 40-store scale this would be the
first thing to add (a Redis layer for hot metrics with 30s TTL).

**Structured logging** uses JSON-formatted log lines so they are parseable by any log aggregator
(Datadog, CloudWatch, etc.) without configuration. Every request logs: trace_id, store_id,
endpoint, latency_ms, status_code.

### Live Dashboard

The dashboard server polls the Intelligence API every 5 seconds and pushes updates to all
connected browsers via WebSocket. This proves the pipeline and API are genuinely connected,
not just batch-processed. The heatmap uses colour intensity (red = high traffic, blue = low)
to make zone activity visually immediate.

---

## AI-Assisted Decisions

### 1. Re-ID approach — HSV histograms vs OSNet

I asked Claude: *"For Re-ID in a retail CCTV context with CPU-only inference and 15fps footage,
what is the best trade-off between accuracy and speed for visitor deduplication?"*

Claude suggested OSNet (torchreid) as the gold standard but noted it requires GPU or would
slow processing to ~0.5fps on CPU. It suggested the histogram approach as a reasonable fallback
for the cross-camera deduplication use case (where the same person is seen on two cameras within
2 seconds) and noted it would struggle for re-entry detection after long gaps.

**I agreed with this assessment** and implemented the histogram approach with a conservative
similarity threshold (0.82) and a separate re-entry window (10 minutes). The limitation is
documented so the team knows where to invest in a GPU upgrade.

### 2. Staff exclusion strategy — model vs heuristic

Claude suggested training a binary classifier on uniform vs non-uniform appearance. I overrode
this in favour of the HSV colour heuristic for two reasons:
1. We have no labelled training data for Purplle's specific uniform
2. The heuristic is explainable and auditable — the team can tune the hue range without
   retraining a model

This was the right call for the current prototype. A production deployment should move to a
fine-tuned classifier once uniform images are collected.

### 3. SQLite vs PostgreSQL for the API

Claude recommended PostgreSQL for production workloads. I chose SQLite for this submission
because the acceptance gate requires `docker compose up` with no manual steps, and SQLite
requires zero external service configuration. The CHOICES.md documents this explicitly.
At 40-store scale with concurrent writes, SQLite would be the first bottleneck.
