# Store Intelligence System — Purplle Tech Challenge 2026

AI-powered store analytics from raw CCTV footage. Converts camera video into a live
Intelligence API with real-time metrics, funnel analysis, anomaly detection, and a
web dashboard.

---

## Quick Start (5 commands)

```bash
# 1. Clone the repository
git clone <your-repo-url> store-intelligence && cd store-intelligence

# 2. Add your footage and data files
cp /path/to/footage/*.mp4 data/footage/
cp /path/to/pos_transactions.csv data/
cp /path/to/store_layout.json data/   # or use the provided one

# 3. Start the API and dashboard
docker compose up --build

# 4. Run the detection pipeline against the clips
bash pipeline/run.sh --api-url http://localhost:8000

# 5. Open the live dashboard
open http://localhost:3000
```

The API is available at `http://localhost:8000`.
The live dashboard is at `http://localhost:3000`.

---

## Running the Detection Pipeline

### Against the CCTV clips (batch mode)
```bash
python pipeline/detect.py \
  --footage-dir data/footage \
  --layout data/store_layout.json \
  --pos data/pos_transactions.csv \
  --output data/events.jsonl \
  --store-id ST1008 \
  --clip-start 2026-04-10T10:00:00Z
```

### Feed output into the API
```bash
# Ingest the generated events into the API
python -c "
import json, requests
events = [json.loads(l) for l in open('data/events.jsonl')]
for i in range(0, len(events), 500):
    r = requests.post('http://localhost:8000/events/ingest',
                      json={'events': events[i:i+500]})
    print(f'Batch {i//500+1}: {r.status_code} — {r.json()[\"accepted\"]} accepted')
"
```

### Real-time mode (for Part E dashboard bonus)
```bash
# Pipeline will POST events directly to API as they are detected
bash pipeline/run.sh --api-url http://localhost:8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/events/ingest` | Ingest batch of events (max 500). Idempotent by event_id. |
| `GET` | `/stores/{id}/metrics` | Real-time KPIs: visitors, conversion rate, dwell, queue |
| `GET` | `/stores/{id}/funnel` | Conversion funnel with drop-off % per stage |
| `GET` | `/stores/{id}/heatmap` | Zone visit frequency + avg dwell, normalised 0–100 |
| `GET` | `/stores/{id}/anomalies` | Active anomalies: queue spike, conversion drop, dead zone |
| `GET` | `/health` | Service health + STALE_FEED warning per store |

**Example:**
```bash
curl http://localhost:8000/stores/ST1008/metrics | python -m json.tool
curl http://localhost:8000/health
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v --cov=app --cov=pipeline --cov-report=term-missing
```

Expected coverage: >70% across `app/` and `pipeline/`.

---

## Project Structure

```
store-intelligence/
├── pipeline/
│   ├── detect.py        # Main detection + tracking (YOLOv8n + ByteTrack)
│   ├── tracker.py       # Re-ID and visitor session management
│   ├── emit.py          # Event schema + emission to file/API
│   └── run.sh           # One-command pipeline runner
├── app/
│   ├── main.py          # FastAPI entrypoint + middleware
│   ├── models.py        # Pydantic event and response schemas
│   ├── database.py      # SQLAlchemy async SQLite setup
│   ├── ingestion.py     # Ingest, deduplication, validation
│   ├── metrics.py       # Real-time metric computation
│   ├── funnel.py        # Conversion funnel logic
│   ├── heatmap.py       # Zone heatmap
│   ├── anomalies.py     # Anomaly detection
│   └── health.py        # Health endpoint
├── dashboard/
│   ├── server.py        # WebSocket dashboard server
│   └── static/
│       └── index.html   # Live web dashboard
├── tests/
│   ├── test_pipeline.py # Pipeline + schema tests (includes AI prompt block)
│   ├── test_metrics.py  # API endpoint tests (idempotency, staff exclusion)
│   └── test_anomalies.py # Anomaly detection tests
├── docs/
│   ├── DESIGN.md        # Architecture + AI-assisted decisions
│   └── CHOICES.md       # 3 key decisions with full reasoning
├── data/
│   ├── store_layout.json   # Zone definitions for ST1008
│   ├── pos_transactions.csv # POS data (derived from real store data)
│   └── footage/            # Place CCTV mp4 files here
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.dashboard
├── requirements.txt
└── README.md
```

---

## Architecture Notes

- **Detection**: YOLOv8n (CPU-optimised) + ByteTrack + HSV colour histogram Re-ID
- **Storage**: SQLite with aiosqlite (swap `DATABASE_URL` for PostgreSQL in production)
- **API**: FastAPI async with structured JSON logging on every request
- **Dashboard**: FastAPI + WebSocket + vanilla HTML/JS (no build step)

See `docs/DESIGN.md` for full architecture and AI-assisted decisions.
See `docs/CHOICES.md` for model selection, schema design, and API architecture reasoning.

---

## Data Sources

This submission uses real Purplle store data:
- Store: **ST1008 Brigade_Bangalore** (Brigade Road, Bangalore)
- POS data: 24 real transactions from 10-Apr-2026 (12:15–21:39)
- CCTV: 5 cameras (CAM1–CAM5), ~680MB total footage

> Note: CCTV footage and POS files are not included in this repository per challenge rules.
