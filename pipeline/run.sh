#!/bin/bash
# One command to process all CCTV clips and feed output into the API
# Usage: bash pipeline/run.sh [--api-url http://localhost:8000]

set -e

FOOTAGE_DIR="${FOOTAGE_DIR:-./data/footage}"
LAYOUT="${LAYOUT:-./data/store_layout.json}"
POS="${POS:-./data/pos_transactions.csv}"
OUTPUT="${OUTPUT:-./data/events.jsonl}"
STORE_ID="${STORE_ID:-ST1008}"
CLIP_START="${CLIP_START:-2026-04-10T10:00:00Z}"
API_URL="${1:-}"

echo "================================================"
echo "  Store Intelligence Detection Pipeline"
echo "================================================"
echo "  Footage : $FOOTAGE_DIR"
echo "  Layout  : $LAYOUT"
echo "  POS     : $POS"
echo "  Output  : $OUTPUT"
echo "  API URL : ${API_URL:-none (batch mode)}"
echo "================================================"

python pipeline/detect.py \
  --footage-dir "$FOOTAGE_DIR" \
  --layout "$LAYOUT" \
  --pos "$POS" \
  --output "$OUTPUT" \
  --store-id "$STORE_ID" \
  --clip-start "$CLIP_START" \
  ${API_URL:+--api-url "$API_URL"}

echo ""
echo "Pipeline complete. Events written to: $OUTPUT"
echo "Event count: $(wc -l < $OUTPUT)"

# If API URL provided, ingest any remaining events
if [ -n "$API_URL" ]; then
  echo "Ingesting events into API at $API_URL..."
  python -c "
import json, requests, sys
url = '$API_URL/events/ingest'
events = [json.loads(l) for l in open('$OUTPUT')]
batch_size = 500
for i in range(0, len(events), batch_size):
    batch = events[i:i+batch_size]
    r = requests.post(url, json={'events': batch}, timeout=30)
    print(f'Batch {i//batch_size+1}: {r.status_code}')
print(f'Done. {len(events)} events ingested.')
"
fi
