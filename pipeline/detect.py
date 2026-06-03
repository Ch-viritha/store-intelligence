"""
Main detection + tracking script.
Processes CCTV clips using YOLOv8 + ByteTrack and emits structured events.

Architecture:
  - YOLOv8n for person detection (fast, good accuracy for retail CCTV)
  - ByteTrack for multi-object tracking across frames (via ultralytics persist=True)
  - Re-ID via HSV colour histogram embeddings (CPU-friendly, no GPU required)
  - Zone classification via bounding box centroid + store_layout.json grid mapping
  - Staff detection via uniform colour heuristic (Purplle green/teal uniform)
  - Cross-camera deduplication via shared visitor_id store keyed on Re-ID embedding
"""

import cv2
import json
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tracker import VisitorTracker
from emit import EventEmitter, build_event

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("detect")

# ── Constants ─────────────────────────────────────────────────────────────────
ENTRY_LINE_RATIO = 0.55         # fraction of frame height for entry/exit threshold line
ZONE_DWELL_INTERVAL_SEC = 30    # emit ZONE_DWELL every N seconds of continuous dwell
CONFIDENCE_THRESHOLD = 0.35     # do NOT suppress events below this — emit and flag
STAFF_HUE_LOW = 35              # HSV hue range for Purplle staff uniform (green/teal)
STAFF_HUE_HIGH = 85
STAFF_SATURATION_MIN = 60
FRAME_SKIP = 3                  # process every 3rd frame (5fps effective at 15fps source)


def load_store_layout(layout_path: str) -> dict:
    with open(layout_path) as f:
        data = json.load(f)
    return {s["store_id"]: s for s in data["stores"]}


def classify_zone(
    cx: float, cy: float,
    frame_w: int, frame_h: int,
    camera_type: str,
    zones: list
) -> Optional[str]:
    """Map bounding box centroid to zone_id based on camera type and normalised position."""
    if camera_type == "entry_exit":
        return None  # entry/exit direction handled separately

    if camera_type == "billing":
        return "BILLING_COUNTER"

    # Floor cameras: divide frame into a grid and map to product zones
    product_zones = [z["zone_id"] for z in zones if z["type"] == "product"]
    if not product_zones:
        return None

    norm_x = cx / max(frame_w, 1)
    norm_y = cy / max(frame_h, 1)
    col = min(int(norm_x * 3), 2)   # 0, 1, 2
    row = min(int(norm_y * 3), 2)   # 0, 1, 2
    idx = min(row * 3 + col, len(product_zones) - 1)
    return product_zones[idx]


def detect_staff(frame: np.ndarray, bbox: tuple) -> tuple:
    """
    Detect staff via dominant uniform colour in the torso region of the bounding box.
    Returns (is_staff: bool, confidence: float).
    Purplle staff wear a distinctive green/teal uniform detectable via HSV masking.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    fh, fw = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(fw, x2); y2 = min(fh, y2)

    if x2 <= x1 or y2 <= y1:
        return False, 0.0

    # Torso = middle third of the bounding box vertically
    torso_y1 = y1 + (y2 - y1) // 3
    torso_y2 = y1 + 2 * (y2 - y1) // 3
    torso = frame[torso_y1:torso_y2, x1:x2]
    if torso.size == 0:
        return False, 0.0

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([STAFF_HUE_LOW, STAFF_SATURATION_MIN, 40]),
        np.array([STAFF_HUE_HIGH, 255, 255])
    )
    total_pixels = mask.shape[0] * mask.shape[1]
    green_ratio = float(np.sum(mask > 0)) / max(total_pixels, 1)
    staff_flag = green_ratio > 0.30
    confidence = float(min(green_ratio * 3.0, 1.0))
    return staff_flag, confidence


def determine_direction(
    prev_cy: Optional[float],
    curr_cy: float,
    entry_line_y: float
) -> Optional[str]:
    """
    Determine ENTRY or EXIT by detecting threshold line crossing.
    Downward movement (increasing y) = customer entering.
    Upward movement (decreasing y) = customer exiting.
    """
    if prev_cy is None:
        return None
    if prev_cy < entry_line_y <= curr_cy:
        return "ENTRY"
    if prev_cy > entry_line_y >= curr_cy:
        return "EXIT"
    return None


def _make_track_state() -> dict:
    return {
        "prev_cy": None,
        "zone_id": None,
        "zone_enter_time": None,
        "dwell_last_emit": None,
        "session_seq": 0,
        "crossed_entry": False,
    }


def process_clip(
    video_path: str,
    store_id: str,
    camera_id: str,
    camera_type: str,
    clip_start_ts: datetime,
    tracker: VisitorTracker,
    emitter: EventEmitter,
    store_zones: list,
    fps_override: Optional[float] = None,
):
    """Process a single CCTV clip and emit structured behavioural events."""
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Cannot open video: {video_path}")
        return

    fps = fps_override or (cap.get(cv2.CAP_PROP_FPS) or 15.0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    entry_line_y = frame_h * ENTRY_LINE_RATIO

    log.info(f"Processing {video_path} | camera={camera_id} | {frame_w}x{frame_h} @ {fps:.1f}fps")

    frame_idx = 0
    track_state: dict = defaultdict(_make_track_state)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            frame_ts = clip_start_ts + timedelta(seconds=frame_idx / fps)

            # YOLOv8 detection + ByteTrack (class 0 = person)
            results = model.track(frame, persist=True, classes=[0], verbose=False)
            if results[0].boxes is None or results[0].boxes.id is None:
                continue

            for box in results[0].boxes:
                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is None:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                visitor_id = tracker.get_or_create_visitor(
                    track_id=track_id,
                    camera_id=camera_id,
                    bbox=(x1, y1, x2, y2),
                    frame=frame,
                    timestamp=frame_ts,
                )

                state = track_state[track_id]
                state["session_seq"] += 1

                staff_flag, staff_conf = detect_staff(frame, (x1, y1, x2, y2))
                # Use max of detection confidence and staff-classification confidence
                detection_conf = max(conf, staff_conf) if staff_flag else conf

                # ── Entry/Exit camera ────────────────────────────────────────
                if camera_type == "entry_exit":
                    direction = determine_direction(state["prev_cy"], cy, entry_line_y)

                    if direction == "ENTRY" and not state["crossed_entry"]:
                        state["crossed_entry"] = True
                        is_reentry = tracker.check_reentry(visitor_id)
                        evt_type = "REENTRY" if is_reentry else "ENTRY"
                        emitter.emit(build_event(
                            store_id=store_id, camera_id=camera_id,
                            visitor_id=visitor_id, event_type=evt_type,
                            timestamp=frame_ts, zone_id=None, dwell_ms=0,
                            is_staff=staff_flag, confidence=detection_conf,
                            session_seq=state["session_seq"],
                        ))
                        tracker.record_entry(visitor_id, frame_ts)

                    elif direction == "EXIT" and state["crossed_entry"]:
                        state["crossed_entry"] = False
                        emitter.emit(build_event(
                            store_id=store_id, camera_id=camera_id,
                            visitor_id=visitor_id, event_type="EXIT",
                            timestamp=frame_ts, zone_id=None, dwell_ms=0,
                            is_staff=staff_flag, confidence=detection_conf,
                            session_seq=state["session_seq"],
                        ))
                        tracker.record_exit(visitor_id, frame_ts)

                # ── Floor / billing cameras ──────────────────────────────────
                else:
                    current_zone = classify_zone(
                        cx, cy, frame_w, frame_h, camera_type, store_zones
                    )

                    if current_zone != state["zone_id"]:
                        # ── Zone exit ────────────────────────────────────────
                        if state["zone_id"] is not None and state["zone_enter_time"] is not None:
                            dwell_ms = int(
                                (frame_ts - state["zone_enter_time"]).total_seconds() * 1000
                            )
                            emitter.emit(build_event(
                                store_id=store_id, camera_id=camera_id,
                                visitor_id=visitor_id, event_type="ZONE_EXIT",
                                timestamp=frame_ts, zone_id=state["zone_id"],
                                dwell_ms=dwell_ms, is_staff=staff_flag,
                                confidence=detection_conf,
                                session_seq=state["session_seq"],
                                sku_zone=state["zone_id"],
                            ))
                            # Billing queue abandon check
                            if state["zone_id"] == "BILLING_COUNTER":
                                q_depth = tracker.get_queue_depth(store_id)
                                if q_depth > 0:
                                    emitter.emit(build_event(
                                        store_id=store_id, camera_id=camera_id,
                                        visitor_id=visitor_id,
                                        event_type="BILLING_QUEUE_ABANDON",
                                        timestamp=frame_ts, zone_id="BILLING_COUNTER",
                                        dwell_ms=dwell_ms, is_staff=staff_flag,
                                        confidence=detection_conf,
                                        session_seq=state["session_seq"],
                                        queue_depth=q_depth,
                                    ))
                                tracker.decrement_queue(store_id)

                        # ── Zone enter ───────────────────────────────────────
                        if current_zone is not None:
                            q_depth = tracker.get_queue_depth(store_id)
                            if current_zone == "BILLING_COUNTER" and q_depth > 0:
                                evt_type = "BILLING_QUEUE_JOIN"
                                emitter.emit(build_event(
                                    store_id=store_id, camera_id=camera_id,
                                    visitor_id=visitor_id, event_type=evt_type,
                                    timestamp=frame_ts, zone_id=current_zone,
                                    dwell_ms=0, is_staff=staff_flag,
                                    confidence=detection_conf,
                                    session_seq=state["session_seq"],
                                    sku_zone=current_zone, queue_depth=q_depth,
                                ))
                            else:
                                emitter.emit(build_event(
                                    store_id=store_id, camera_id=camera_id,
                                    visitor_id=visitor_id, event_type="ZONE_ENTER",
                                    timestamp=frame_ts, zone_id=current_zone,
                                    dwell_ms=0, is_staff=staff_flag,
                                    confidence=detection_conf,
                                    session_seq=state["session_seq"],
                                    sku_zone=current_zone,
                                ))

                            if current_zone == "BILLING_COUNTER":
                                tracker.increment_queue(store_id)

                        state["zone_id"] = current_zone
                        state["zone_enter_time"] = frame_ts
                        state["dwell_last_emit"] = frame_ts

                    # ── Zone dwell — emit every 30 s of continuous presence ──
                    elif (
                        current_zone is not None
                        and state["dwell_last_emit"] is not None
                        and state["zone_enter_time"] is not None
                    ):
                        elapsed = (frame_ts - state["dwell_last_emit"]).total_seconds()
                        if elapsed >= ZONE_DWELL_INTERVAL_SEC:
                            total_dwell_ms = int(
                                (frame_ts - state["zone_enter_time"]).total_seconds() * 1000
                            )
                            emitter.emit(build_event(
                                store_id=store_id, camera_id=camera_id,
                                visitor_id=visitor_id, event_type="ZONE_DWELL",
                                timestamp=frame_ts, zone_id=current_zone,
                                dwell_ms=total_dwell_ms, is_staff=staff_flag,
                                confidence=detection_conf,
                                session_seq=state["session_seq"],
                                sku_zone=current_zone,
                            ))
                            state["dwell_last_emit"] = frame_ts

                state["prev_cy"] = cy

    finally:
        cap.release()

    log.info(
        f"Finished {video_path} — {frame_idx} frames processed, "
        f"{emitter.count} events emitted total"
    )


def main():
    parser = argparse.ArgumentParser(description="Store Intelligence Detection Pipeline")
    parser.add_argument("--footage-dir", required=True, help="Directory containing CCTV mp4 files")
    parser.add_argument("--layout", required=True, help="Path to store_layout.json")
    parser.add_argument("--pos", required=True, help="Path to pos_transactions.csv")
    parser.add_argument("--output", required=True, help="Output .jsonl file for events")
    parser.add_argument("--store-id", default="STORE_BLR_002")
    parser.add_argument("--clip-start", default="2026-04-10T10:00:00Z",
                        help="ISO-8601 UTC start time of clips")
    parser.add_argument("--api-url", default=None,
                        help="If set, POST events to Intelligence API in real-time")
    args = parser.parse_args()

    stores = load_store_layout(args.layout)
    store = stores.get(args.store_id)
    if not store:
        log.error(f"Store '{args.store_id}' not found in {args.layout}")
        sys.exit(1)

    clip_start = datetime.fromisoformat(args.clip_start.replace("Z", "+00:00"))
    tracker = VisitorTracker(pos_csv=args.pos)
    emitter = EventEmitter(output_path=args.output, api_url=args.api_url)

    footage_dir = Path(args.footage_dir)
    for cam_info in store["cameras"]:
        video_path = footage_dir / cam_info["file"]
        if not video_path.exists():
            log.warning(f"Video not found: {video_path} — skipping")
            continue

        process_clip(
            video_path=str(video_path),
            store_id=args.store_id,
            camera_id=cam_info["camera_id"],
            camera_type=cam_info["type"],
            clip_start_ts=clip_start,
            tracker=tracker,
            emitter=emitter,
            store_zones=store["zones"],
        )

    emitter.flush()
    log.info(f"Pipeline complete. Total events: {emitter.count} → {args.output}")


if __name__ == "__main__":
    main()
