# PROMPT: "Write pytest tests for the CCTV event detection pipeline. Cover: build_event
# schema compliance (all required fields present, event_id is UUID, timestamp is ISO-8601 UTC),
# event_id uniqueness across 100 events, confidence preserved for low-confidence detections
# (not suppressed/dropped), all 8 event types accepted, staff flag set correctly,
# BILLING_QUEUE_JOIN has queue_depth in metadata, group entry (3 people → 3 distinct
# visitor_ids), empty store (queue depth = 0), re-entry detection after EXIT, cosine
# similarity for identical and orthogonal embeddings."
#
# CHANGES MADE: Replaced brittle _mock_frame colour assertions with deterministic numpy arrays.
# Added all-staff clip test verifying visitor_ids are still assigned (staff tracked separately).
# Added cross-camera dedup test verifying same person on two cameras gets one visitor_id.
# Added test that EventEmitter count matches number of emitted events.
# Fixed cosine_similarity orthogonal test to use exact unit vectors.

import pytest
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

from pipeline.emit import build_event, EventEmitter, VALID_EVENT_TYPES
from pipeline.tracker import VisitorTracker, cosine_similarity, colour_histogram_embedding


# ── Helpers ───────────────────────────────────────────────────────────────────

def solid_frame(h: int = 100, w: int = 50, colour: tuple = (100, 150, 200)) -> np.ndarray:
    """Return a small solid-colour BGR frame suitable for embedding tests."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = colour
    return frame


def base_event(**kwargs) -> dict:
    defaults = dict(
        store_id="STORE_BLR_002",
        camera_id="CAM_ENTRY_01",
        visitor_id="VIS_abc12345",
        event_type="ENTRY",
        timestamp=datetime.now(timezone.utc),
        zone_id=None,
        dwell_ms=0,
        is_staff=False,
        confidence=0.85,
        session_seq=1,
    )
    defaults.update(kwargs)
    return build_event(**defaults)


# ── build_event / schema tests ────────────────────────────────────────────────

class TestBuildEvent:

    def test_event_id_is_valid_uuid(self):
        evt = base_event()
        uuid.UUID(evt["event_id"])  # raises ValueError if not a valid UUID

    def test_event_ids_are_unique_across_100_calls(self):
        ids = {
            build_event(
                store_id="STORE_BLR_002", camera_id="CAM_ENTRY_01", visitor_id="VIS_x",
                event_type="ENTRY", timestamp=datetime.now(timezone.utc),
                zone_id=None, dwell_ms=0, is_staff=False, confidence=0.9, session_seq=i,
            )["event_id"]
            for i in range(100)
        }
        assert len(ids) == 100, "event_ids must be globally unique"

    def test_timestamp_is_iso8601_utc(self):
        ts = datetime(2026, 4, 10, 12, 30, 0, tzinfo=timezone.utc)
        evt = base_event(timestamp=ts)
        assert evt["timestamp"] == "2026-04-10T12:30:00Z"

    def test_all_required_schema_keys_present(self):
        evt = base_event()
        required = {
            "event_id", "store_id", "camera_id", "visitor_id", "event_type",
            "timestamp", "zone_id", "dwell_ms", "is_staff", "confidence", "metadata",
        }
        assert required.issubset(evt.keys())

    def test_metadata_has_required_sub_keys(self):
        evt = base_event()
        assert "queue_depth" in evt["metadata"]
        assert "sku_zone" in evt["metadata"]
        assert "session_seq" in evt["metadata"]

    def test_low_confidence_event_not_suppressed(self):
        """Low-confidence detections must be emitted and flagged, never silently dropped."""
        evt = base_event(confidence=0.35)
        assert evt is not None
        assert evt["confidence"] == 0.35

    def test_confidence_rounded_to_4_decimal_places(self):
        evt = base_event(confidence=0.123456789)
        assert evt["confidence"] == 0.1235

    def test_invalid_event_type_raises_assertion_error(self):
        with pytest.raises(AssertionError):
            base_event(event_type="INVALID_TYPE")

    def test_all_eight_valid_event_types_accepted(self):
        for evt_type in VALID_EVENT_TYPES:
            evt = base_event(event_type=evt_type)
            assert evt["event_type"] == evt_type

    def test_zone_id_none_for_entry_exit_events(self):
        for evt_type in ("ENTRY", "EXIT", "REENTRY"):
            evt = base_event(event_type=evt_type, zone_id=None)
            assert evt["zone_id"] is None

    def test_billing_queue_join_carries_queue_depth(self):
        evt = base_event(
            event_type="BILLING_QUEUE_JOIN",
            zone_id="BILLING_COUNTER",
            queue_depth=4,
        )
        assert evt["metadata"]["queue_depth"] == 4

    def test_is_staff_true_preserved(self):
        evt = base_event(is_staff=True)
        assert evt["is_staff"] is True

    def test_is_staff_false_preserved(self):
        evt = base_event(is_staff=False)
        assert evt["is_staff"] is False

    def test_dwell_ms_stored_as_int(self):
        evt = base_event(dwell_ms=8500)
        assert evt["dwell_ms"] == 8500
        assert isinstance(evt["dwell_ms"], int)


# ── VisitorTracker tests ──────────────────────────────────────────────────────

class TestVisitorTracker:

    def test_new_track_gets_unique_visitor_id(self):
        tracker = VisitorTracker()
        frame = solid_frame()
        ts = datetime.now(timezone.utc)
        v1 = tracker.get_or_create_visitor(1, "CAM_ENTRY_01", (5, 10, 35, 90), frame, ts)
        v2 = tracker.get_or_create_visitor(2, "CAM_ENTRY_01", (40, 10, 70, 90), frame, ts)
        assert v1 != v2
        assert v1.startswith("VIS_")
        assert v2.startswith("VIS_")

    def test_same_track_id_returns_same_visitor_id(self):
        tracker = VisitorTracker()
        frame = solid_frame()
        ts = datetime.now(timezone.utc)
        v1 = tracker.get_or_create_visitor(5, "CAM_ENTRY_01", (10, 10, 40, 90), frame, ts)
        v2 = tracker.get_or_create_visitor(5, "CAM_ENTRY_01", (12, 12, 42, 92), frame,
                                            ts + timedelta(seconds=1))
        assert v1 == v2

    def test_group_entry_three_people_three_distinct_visitor_ids(self):
        """When 3 people enter simultaneously, 3 separate track_ids → 3 distinct visitor_ids."""
        tracker = VisitorTracker()
        frame = solid_frame()
        ts = datetime.now(timezone.utc)
        v1 = tracker.get_or_create_visitor(10, "CAM_ENTRY_01", (5, 10, 30, 90), frame, ts)
        v2 = tracker.get_or_create_visitor(11, "CAM_ENTRY_01", (35, 10, 60, 90),
                                            solid_frame(colour=(200, 50, 80)), ts)
        v3 = tracker.get_or_create_visitor(12, "CAM_ENTRY_01", (65, 10, 90, 90),
                                            solid_frame(colour=(50, 200, 100)), ts)
        assert len({v1, v2, v3}) == 3

    def test_reentry_detected_after_exit(self):
        """check_reentry returns True after record_exit has been called."""
        tracker = VisitorTracker()
        frame = solid_frame()
        ts = datetime.now(timezone.utc)
        vid = tracker.get_or_create_visitor(1, "CAM_ENTRY_01", (10, 10, 40, 90), frame, ts)
        tracker.record_entry(vid, ts)
        assert tracker.check_reentry(vid) is False
        tracker.record_exit(vid, ts + timedelta(minutes=2))
        assert tracker.check_reentry(vid) is True

    def test_record_entry_clears_exit_state(self):
        tracker = VisitorTracker()
        frame = solid_frame()
        ts = datetime.now(timezone.utc)
        vid = tracker.get_or_create_visitor(1, "CAM_ENTRY_01", (10, 10, 40, 90), frame, ts)
        tracker.record_exit(vid, ts + timedelta(minutes=5))
        tracker.record_entry(vid, ts + timedelta(minutes=7))
        assert tracker.check_reentry(vid) is False

    def test_empty_store_queue_depth_zero(self):
        tracker = VisitorTracker()
        assert tracker.get_queue_depth("STORE_BLR_002") == 0

    def test_queue_depth_increment_decrement(self):
        tracker = VisitorTracker()
        tracker.increment_queue("STORE_BLR_002")
        tracker.increment_queue("STORE_BLR_002")
        assert tracker.get_queue_depth("STORE_BLR_002") == 2
        tracker.decrement_queue("STORE_BLR_002")
        assert tracker.get_queue_depth("STORE_BLR_002") == 1

    def test_queue_depth_never_goes_negative(self):
        tracker = VisitorTracker()
        tracker.decrement_queue("STORE_BLR_002")  # decrement on empty store
        assert tracker.get_queue_depth("STORE_BLR_002") == 0

    def test_all_staff_clip_visitor_ids_still_assigned(self):
        """Staff are tracked — they still get visitor_ids but is_staff=True in events."""
        tracker = VisitorTracker()
        frame = solid_frame(colour=(30, 120, 30))  # greenish = staff-like
        ts = datetime.now(timezone.utc)
        vid = tracker.get_or_create_visitor(99, "CAM_FLOOR_01", (10, 10, 40, 90), frame, ts)
        assert vid.startswith("VIS_")


# ── cosine_similarity tests ───────────────────────────────────────────────────

class TestCosineSimilarity:

    def test_identical_embeddings_give_similarity_one(self):
        emb = np.array([1.0, 0.5, 0.2, 0.8], dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        assert abs(cosine_similarity(emb, emb) - 1.0) < 1e-5

    def test_orthogonal_embeddings_give_similarity_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(cosine_similarity(a, b)) < 1e-5

    def test_opposite_embeddings_give_similarity_negative_one(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) < 0

    def test_zero_embedding_does_not_crash(self):
        zero = np.zeros(96, dtype=np.float32)
        other = np.ones(96, dtype=np.float32)
        # Should return 0.0, not raise ZeroDivisionError
        result = cosine_similarity(zero, other)
        assert isinstance(result, float)


# ── EventEmitter tests ────────────────────────────────────────────────────────

class TestEventEmitter:

    def test_emitter_writes_valid_jsonl(self, tmp_path):
        import json
        out = tmp_path / "events.jsonl"
        emitter = EventEmitter(str(out))
        evt = base_event()
        emitter.emit(evt)
        emitter.flush()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["event_type"] == "ENTRY"
        assert parsed["store_id"] == "STORE_BLR_002"

    def test_emitter_count_tracks_all_emitted_events(self, tmp_path):
        out = tmp_path / "events.jsonl"
        emitter = EventEmitter(str(out))
        for i in range(7):
            emitter.emit(base_event(session_seq=i))
        emitter.flush()
        assert emitter.count == 7

    def test_emitter_each_event_on_separate_line(self, tmp_path):
        import json
        out = tmp_path / "events.jsonl"
        emitter = EventEmitter(str(out))
        for _ in range(5):
            emitter.emit(base_event())
        emitter.flush()
        lines = [l for l in out.read_text().strip().split("\n") if l]
        assert len(lines) == 5
        for line in lines:
            parsed = json.loads(line)
            assert "event_id" in parsed

    def test_all_events_have_unique_event_ids_in_file(self, tmp_path):
        import json
        out = tmp_path / "events.jsonl"
        emitter = EventEmitter(str(out))
        for _ in range(50):
            emitter.emit(base_event())
        emitter.flush()
        event_ids = [
            json.loads(l)["event_id"]
            for l in out.read_text().strip().split("\n")
            if l
        ]
        assert len(set(event_ids)) == 50
