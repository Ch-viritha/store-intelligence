"""
Re-ID and visitor session tracking logic.

Design decisions:
  - visitor_id is scoped to a session, not a person permanently
  - Re-entry: same appearance embedding returning after EXIT within 10 min → REENTRY event
  - Cross-camera dedup: same embedding seen on overlapping camera within 2s → same visitor_id
  - Staff tracked but excluded from customer metrics via is_staff flag at API layer
  - Queue depth tracked per store as a simple counter (incremented on enter, decremented on exit)
"""

import csv
import uuid
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, List

import numpy as np

log = logging.getLogger("tracker")

REENTRY_WINDOW_SEC = 600        # 10 min — person returning within this window = REENTRY
CROSS_CAM_WINDOW_SEC = 2        # 2s — same person on overlapping cameras = same visitor
SIMILARITY_THRESHOLD = 0.82     # cosine similarity required for Re-ID match


def colour_histogram_embedding(frame: np.ndarray, bbox: tuple, bins: int = 32) -> np.ndarray:
    """
    Compute an HSV colour histogram embedding for Re-ID.
    96-dimensional vector (32 bins × H, S, V channels), L2-normalised.
    CPU-friendly alternative to OSNet — works well for cross-camera dedup
    and short-window re-entry detection.
    """
    import cv2

    x1, y1, x2, y2 = [int(v) for v in bbox]
    fh, fw = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(fw, x2); y2 = min(fh, y2)

    if x2 <= x1 or y2 <= y1:
        return np.zeros(bins * 3, dtype=np.float32)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros(bins * 3, dtype=np.float32)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()

    embedding = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    norm = np.linalg.norm(embedding)
    return embedding / (norm + 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / denom)


class VisitorTracker:
    def __init__(self, pos_csv: Optional[str] = None):
        # (camera_id, track_id) → visitor_id
        self._track_to_visitor: dict = {}
        # visitor_id → metadata dict
        self._visitor_store: dict = {}
        # store_id → queue depth (integer counter)
        self._queue_depth: dict = defaultdict(int)
        # POS transactions for conversion correlation
        self._pos_transactions: List[dict] = self._load_pos(pos_csv) if pos_csv else []

    # ── POS loading ───────────────────────────────────────────────────────────

    def _load_pos(self, csv_path: str) -> List[dict]:
        transactions = []
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = datetime.fromisoformat(
                            row["timestamp"].strip().replace("Z", "+00:00")
                        )
                        transactions.append({
                            "store_id": row["store_id"].strip(),
                            "transaction_id": row["transaction_id"].strip(),
                            "timestamp": ts,
                            "basket_value_inr": float(row.get("basket_value_inr") or 0),
                        })
                    except (KeyError, ValueError):
                        continue
        except FileNotFoundError:
            log.warning(f"POS CSV not found: {csv_path}")
        log.info(f"Loaded {len(transactions)} POS transactions from {csv_path}")
        return transactions

    def get_pos_in_window(
        self, store_id: str, window_start: datetime, window_end: datetime
    ) -> List[dict]:
        """Return POS transactions for a store within a time window."""
        return [
            t for t in self._pos_transactions
            if t["store_id"] == store_id
            and window_start <= t["timestamp"] <= window_end
        ]

    # ── Visitor ID assignment ─────────────────────────────────────────────────

    def get_or_create_visitor(
        self,
        track_id: int,
        camera_id: str,
        bbox: tuple,
        frame: np.ndarray,
        timestamp: datetime,
    ) -> str:
        key = (camera_id, track_id)

        if key in self._track_to_visitor:
            vid = self._track_to_visitor[key]
            info = self._visitor_store[vid]
            info["last_seen"] = timestamp
            info["camera_id"] = camera_id
            info["embedding"] = colour_histogram_embedding(frame, bbox)
            return vid

        embedding = colour_histogram_embedding(frame, bbox)
        matched_vid = self._find_match(embedding, timestamp, camera_id)

        if matched_vid:
            self._track_to_visitor[key] = matched_vid
            info = self._visitor_store[matched_vid]
            info["last_seen"] = timestamp
            info["camera_id"] = camera_id
            info["embedding"] = embedding
            return matched_vid

        # Assign a new visitor_id
        vid = "VIS_" + uuid.uuid4().hex[:8]
        self._track_to_visitor[key] = vid
        self._visitor_store[vid] = {
            "embedding": embedding,
            "entry_time": timestamp,
            "last_seen": timestamp,
            "exit_time": None,
            "camera_id": camera_id,
            "has_exited": False,
        }
        return vid

    def _find_match(
        self, embedding: np.ndarray, timestamp: datetime, camera_id: str
    ) -> Optional[str]:
        """
        Find an existing visitor matching by cosine similarity.
        Checks for cross-camera dedup (same person, different camera, within 2s)
        and re-entry (same person returning after EXIT within 10 min).
        """
        best_vid = None
        best_sim = SIMILARITY_THRESHOLD

        for vid, info in self._visitor_store.items():
            stored_emb = info.get("embedding")
            if stored_emb is None:
                continue

            sim = cosine_similarity(embedding, stored_emb)
            if sim <= best_sim:
                continue

            last_seen = info["last_seen"]
            # Ensure consistent timezone handling
            if last_seen.tzinfo is None and timestamp.tzinfo is not None:
                last_seen = last_seen.replace(tzinfo=timestamp.tzinfo)
            elif last_seen.tzinfo is not None and timestamp.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=None)

            elapsed = (timestamp - last_seen).total_seconds()

            # Cross-camera dedup: same appearance on a different camera within 2s
            if elapsed <= CROSS_CAM_WINDOW_SEC and info["camera_id"] != camera_id:
                best_sim = sim
                best_vid = vid
                continue

            # Re-entry: person who exited and returns within 10 minutes
            if info["has_exited"] and info["exit_time"] is not None:
                exit_time = info["exit_time"]
                if exit_time.tzinfo is None and timestamp.tzinfo is not None:
                    exit_time = exit_time.replace(tzinfo=timestamp.tzinfo)
                elif exit_time.tzinfo is not None and timestamp.tzinfo is None:
                    exit_time = exit_time.replace(tzinfo=None)

                reentry_elapsed = (timestamp - exit_time).total_seconds()
                if 0 <= reentry_elapsed <= REENTRY_WINDOW_SEC:
                    best_sim = sim
                    best_vid = vid

        return best_vid

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def check_reentry(self, visitor_id: str) -> bool:
        """Return True if this visitor has previously exited (i.e. this is a re-entry)."""
        return self._visitor_store.get(visitor_id, {}).get("has_exited", False)

    def record_entry(self, visitor_id: str, timestamp: datetime):
        info = self._visitor_store.get(visitor_id)
        if info:
            info["entry_time"] = timestamp
            info["has_exited"] = False
            info["exit_time"] = None

    def record_exit(self, visitor_id: str, timestamp: datetime):
        info = self._visitor_store.get(visitor_id)
        if info:
            info["has_exited"] = True
            info["exit_time"] = timestamp

    # ── Queue depth ───────────────────────────────────────────────────────────

    def get_queue_depth(self, store_id: str) -> int:
        return self._queue_depth[store_id]

    def increment_queue(self, store_id: str):
        self._queue_depth[store_id] += 1

    def decrement_queue(self, store_id: str):
        self._queue_depth[store_id] = max(0, self._queue_depth[store_id] - 1)
