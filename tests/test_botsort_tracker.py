"""Integration tests for BoTSORTTracker.

These drive the *real* Ultralytics BoT-SORT tracker, but feed it synthetic
detections — no YOLO model, no weights and no GPU. GMC is set to "none"
so the association is deterministic on blank frames.
"""

from __future__ import annotations

import numpy as np
import pytest

from detection.data_models import Detection
from tracking.botsort_tracker import BoTSORTTracker
from utils.config_loader import BallTrackingConfig, BotSortParams, TrackingConfig

CLASS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


def make_config(separate_ball=True):
    """Tracking config with GMC disabled for deterministic tests.

    These tests exercise the BoT-SORT ball *fallback* path, so the custom
    BallTracker is switched off here (it has its own test module).
    """
    main = BotSortParams(gmc_method="none")
    ball = BotSortParams(
        track_high_thresh=0.1,
        track_low_thresh=0.05,
        new_track_thresh=0.1,
        track_buffer=60,
        fuse_score=False,
        gmc_method="none",
    )
    return TrackingConfig(
        enabled=True,
        separate_ball=separate_ball,
        main=main,
        ball=ball,
        ball_tracking=BallTrackingConfig(enabled=False),
    )


def det(class_id, x1, y1, x2, y2, conf=0.9):
    return Detection(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        confidence=conf,
        bbox=(x1, y1, x2, y2),
    )


def make_tracker(separate_ball=True):
    return BoTSORTTracker(make_config(separate_ball), CLASS_NAMES, frame_rate=30)


def test_single_player_keeps_one_id_while_moving():
    tracker = make_tracker()
    ids = set()
    for i in range(6):
        x = 100 + i * 6  # drift right a few px per frame
        tracks = tracker.update([det(2, x, 100, x + 40, 200)], FRAME, i + 1)
        assert len(tracks) == 1
        assert tracks[0].class_name == "player"
        ids.add(tracks[0].track_id)
    assert len(ids) == 1, f"player ID should be stable, saw {ids}"


def test_two_players_get_two_distinct_stable_ids():
    tracker = make_tracker()
    seen = {"a": set(), "b": set()}
    for i in range(5):
        a_x = 100 + i * 5
        b_x = 600 - i * 5
        tracks = tracker.update(
            [det(2, a_x, 100, a_x + 40, 200), det(2, b_x, 400, b_x + 40, 500)],
            FRAME, i + 1,
        )
        assert len(tracks) == 2
        # Left player has smaller x-center than right player.
        left, right = sorted(tracks, key=lambda t: t.bbox[0])
        seen["a"].add(left.track_id)
        seen["b"].add(right.track_id)
    assert len(seen["a"]) == 1 and len(seen["b"]) == 1
    assert seen["a"] != seen["b"], "the two players must have different IDs"


def test_ball_tracked_by_separate_tracker():
    tracker = make_tracker(separate_ball=True)
    ball_ids = set()
    for i in range(5):
        x = 300 + i * 3
        tracks = tracker.update(
            [det(2, 100, 100, 140, 200), det(0, x, 500, x + 12, 512, conf=0.3)],
            FRAME, i + 1,
        )
        ball_tracks = [t for t in tracks if t.class_name == "ball"]
        assert len(ball_tracks) == 1
        ball_ids.add(ball_tracks[0].track_id)
    assert len(ball_ids) == 1


def test_source_detection_id_points_at_original_index():
    tracker = make_tracker(separate_ball=True)
    # Frame layout: index 0 = ball, index 1 = player.
    detections = [
        det(0, 300, 500, 312, 512, conf=0.3),
        det(2, 100, 100, 140, 200),
    ]
    tracks = tracker.update(detections, FRAME, 1)
    by_class = {t.class_name: t for t in tracks}

    assert by_class["player"].source_detection_id == 1
    assert by_class["ball"].source_detection_id == 0


def test_empty_frame_returns_no_tracks_and_does_not_crash():
    tracker = make_tracker()
    assert tracker.update([], FRAME, 1) == []
    # A track that first appears after frame 1 needs a second consecutive
    # frame to be confirmed (standard ByteTrack/BoT-SORT behaviour).
    tracker.update([det(2, 100, 100, 140, 200)], FRAME, 2)
    tracks = tracker.update([det(2, 103, 100, 143, 200)], FRAME, 3)
    assert len(tracks) == 1
    assert tracks[0].class_name == "player"


def test_disappearing_object_is_not_emitted_when_absent():
    tracker = make_tracker()
    tracker.update([det(2, 100, 100, 140, 200)], FRAME, 1)
    # No detections this frame -> coasting track is not output.
    assert tracker.update([], FRAME, 2) == []


def test_reid_auto_falls_back_without_crashing():
    config = make_config()
    config.with_reid = True
    config.reid_model = "auto"  # unsupported standalone -> must fall back
    tracker = BoTSORTTracker(config, CLASS_NAMES, frame_rate=30)
    tracks = tracker.update([det(2, 100, 100, 140, 200)], FRAME, 1)
    assert len(tracks) == 1


def test_reset_runs_clean():
    tracker = make_tracker()
    tracker.update([det(2, 100, 100, 140, 200)], FRAME, 1)
    tracker.reset()  # should not raise


def test_custom_ball_tracker_handles_zero_iou_ball_without_touching_players():
    # ball_tracking enabled (default): the small/fast ball is routed to the
    # custom BallTracker while players still go through BoT-SORT.
    cfg = TrackingConfig(
        enabled=True,
        separate_ball=True,
        main=BotSortParams(gmc_method="none"),
        ball_tracking=BallTrackingConfig(enabled=True),
    )
    tracker = BoTSORTTracker(cfg, CLASS_NAMES, frame_rate=30)
    real_ball = {
        18: (280.25, 241.38, 286.48, 251.29),
        19: (270.78, 241.33, 277.70, 251.18),
        20: (263.14, 241.67, 269.81, 251.30),
        21: (255.97, 242.52, 262.18, 251.30),
    }
    ball_ids, player_ids = set(), set()
    for f in (18, 19, 20, 21):
        x = 100 + (f - 18) * 5
        tracks = tracker.update(
            [det(2, x, 100, x + 40, 200), det(0, *real_ball[f], conf=0.35)],
            FRAME, f,
        )
        for t in tracks:
            (ball_ids if t.class_name == "ball" else player_ids).add(t.track_id)

    # The ball got a stable, offset ID despite zero raw IoU.
    assert len(ball_ids) == 1
    assert next(iter(ball_ids)) > 9000
    # The player kept a single stable ID too.
    assert len(player_ids) == 1
