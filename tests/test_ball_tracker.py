"""Tests for the custom BallTracker (no Ultralytics, no GPU).

The headline test replays the *real* ball coordinates from input_video2
frames 18-21 — the ones that have zero IoU between consecutive frames and
that BoT-SORT failed to associate.
"""

from __future__ import annotations

import pytest

from detection.data_models import Detection
from tracking.ball_tracker import _inflate, _iou, BallTracker
from utils.config_loader import BallTrackingConfig

BALL_CLASS_ID = 0

# Real ball boxes from input_video2 (consecutive, zero raw IoU).
REAL_BALL = {
    18: (280.25, 241.38, 286.48, 251.29),
    19: (270.78, 241.33, 277.70, 251.18),
    20: (263.14, 241.67, 269.81, 251.30),
    21: (255.97, 242.52, 262.18, 253.01),
}


def ball_det(bbox, conf=0.35):
    return Detection(BALL_CLASS_ID, "ball", conf, bbox)


def make_tracker(**overrides):
    config = BallTrackingConfig(**overrides)
    return BallTracker(config, BALL_CLASS_ID, "ball")


def test_raw_consecutive_ball_boxes_have_zero_iou():
    # This is the whole reason the custom tracker exists.
    assert _iou(REAL_BALL[18], REAL_BALL[19]) == 0.0
    assert _iou(REAL_BALL[19], REAL_BALL[20]) == 0.0


def test_inflation_restores_overlap():
    a = _inflate(REAL_BALL[18], scale=3.0, min_size=20.0)
    b = _inflate(REAL_BALL[19], scale=3.0, min_size=20.0)
    assert _iou(a, b) > 0.0


def test_real_zero_iou_ball_gets_persistent_id():
    tracker = make_tracker()
    results = {}
    for frame in (18, 19, 20, 21):
        results[frame] = tracker.update([(0, ball_det(REAL_BALL[frame]))], frame)

    # Frame 18 is the cold start: one detection, not yet confirmed.
    assert results[18] == []
    # Frames 19-21: confirmed and emitted with a single, stable ID.
    ids = set()
    for frame in (19, 20, 21):
        assert len(results[frame]) == 1, f"frame {frame}"
        assert results[frame][0].class_name == "ball"
        ids.add(results[frame][0].track_id)
    assert len(ids) == 1, f"ball ID must be stable, saw {ids}"
    assert next(iter(ids)) > 9000, "ball IDs must be offset from player IDs"


def test_exported_bbox_is_the_true_box_not_inflated():
    tracker = make_tracker()
    tracker.update([(0, ball_det(REAL_BALL[18]))], 18)
    tracks = tracker.update([(0, ball_det(REAL_BALL[19]))], 19)
    assert tracks[0].bbox == REAL_BALL[19]


def test_source_detection_id_is_preserved():
    tracker = make_tracker()
    tracker.update([(5, ball_det(REAL_BALL[18]))], 18)
    tracks = tracker.update([(7, ball_det(REAL_BALL[19]))], 19)
    assert tracks[0].source_detection_id == 7


def test_single_detection_is_not_emitted():
    tracker = make_tracker()
    assert tracker.update([(0, ball_det(REAL_BALL[18]))], 18) == []


def test_distance_gate_matches_when_iou_is_zero():
    # Boxes 30 px apart: even inflated (20 px) they don't overlap, so the
    # match must come from the center-distance gate (30 < 50).
    tracker = make_tracker(distance_gate_px=50.0)
    tracker.update([(0, ball_det((100, 100, 106, 110)))], 1)
    tracks = tracker.update([(0, ball_det((130, 100, 136, 110)))], 2)
    assert len(tracks) == 1


def test_beyond_distance_gate_creates_new_track():
    tracker = make_tracker(distance_gate_px=20.0)
    tracker.update([(0, ball_det((100, 100, 106, 110)))], 1)
    tracker.update([(0, ball_det((100, 100, 106, 110)))], 2)  # confirm id A
    # Jump 200 px away, far beyond gate and inflated IoU -> new track.
    tracker.update([(0, ball_det((300, 100, 306, 110)))], 3)
    tracks = tracker.update([(0, ball_det((300, 100, 306, 110)))], 4)
    assert len(tracks) == 1
    # Different identity than the first ball.
    assert tracks[0].track_id == 9002


def test_velocity_prediction_bridges_a_one_frame_gap():
    tracker = make_tracker(max_frame_gap=5, distance_gate_px=15.0)
    # Establish a steady leftward motion of ~9 px/frame.
    tracker.update([(0, ball_det((280, 241, 286, 251)))], 1)
    tracker.update([(0, ball_det((271, 241, 277, 251)))], 2)  # confirmed
    # Frame 3 has NO ball. Frame 4 reappears where velocity predicts
    # (~262..253 -> center jumps ~18 px over 2 frames). Without prediction
    # this would be a 18 px center jump and fail a 15 px gate.
    tracks = tracker.update([(0, ball_det((253, 241, 259, 251)))], 4)
    assert len(tracks) == 1, "velocity prediction should re-acquire the ball"


def test_stale_track_dropped_after_max_frame_gap():
    tracker = make_tracker(max_frame_gap=2)
    tracker.update([(0, ball_det((280, 241, 286, 251)))], 1)
    tracker.update([(0, ball_det((271, 241, 277, 251)))], 2)  # id 9001, confirmed
    # Large gap (> max_frame_gap) -> old track dropped, new identity.
    tracks = tracker.update([(0, ball_det((100, 500, 106, 510)))], 10)
    tracks = tracker.update([(0, ball_det((100, 500, 106, 510)))], 11)
    assert tracks[0].track_id != 9001


def test_reset_clears_state():
    tracker = make_tracker()
    tracker.update([(0, ball_det(REAL_BALL[18]))], 18)
    tracker.reset()
    # After reset the ID counter restarts.
    tracker.update([(0, ball_det(REAL_BALL[18]))], 1)
    tracks = tracker.update([(0, ball_det(REAL_BALL[19]))], 2)
    assert tracks[0].track_id == 9001
