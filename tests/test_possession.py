"""Tests for ball-possession analytics (image-space, no homography)."""

from __future__ import annotations

import json

from analytics.possession import (
    compute_possession,
    export_possession,
    frame_possessor,
    smooth_possession,
)
from tracking.models import FrameTracks, Track


def _track(tid, name, cx, cy, w=20, h=60):
    return Track(1, tid, 2, name, 0.9, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def _frame(idx, tracks):
    return FrameTracks(idx, tracks)


# ---------------------------------------------------------------------------
# Per-frame possessor
# ---------------------------------------------------------------------------
def test_frame_possessor_picks_nearest_player_team():
    # Ball at x=500; team-0 player at 505 (closer), team-1 player at 700.
    tracks = [
        _track(99, "ball", 500, 400, w=8, h=8),
        _track(1, "player", 505, 380),
        _track(2, "player", 700, 380),
    ]
    roles = {1: ("player", 0), 2: ("player", 1)}
    team, pid = frame_possessor(_frame(1, tracks), roles, "ball", gate_factor=3.0)
    assert team == 0 and pid == 1


def test_frame_possessor_none_when_ball_far():
    tracks = [
        _track(99, "ball", 100, 100, w=8, h=8),
        _track(1, "player", 900, 600),     # far from the ball
    ]
    roles = {1: ("player", 0)}
    assert frame_possessor(_frame(1, tracks), roles, "ball", gate_factor=1.0) is None


def test_frame_possessor_none_without_ball():
    tracks = [_track(1, "player", 500, 400)]
    assert frame_possessor(_frame(1, tracks), {1: ("player", 0)}, "ball") is None


def test_frame_possessor_ignores_referee():
    tracks = [
        _track(99, "ball", 500, 400, w=8, h=8),
        _track(5, "referee", 502, 380),    # closest, but a referee can't possess
        _track(1, "player", 540, 380),
    ]
    roles = {5: ("referee", None), 1: ("player", 1)}
    team, pid = frame_possessor(_frame(1, tracks), roles, "ball", gate_factor=4.0)
    assert team == 1 and pid == 1


# ---------------------------------------------------------------------------
# Hysteresis / last-touch smoothing
# ---------------------------------------------------------------------------
def test_smooth_possession_last_touch_carries_through_none():
    raw = [0, 0, None, None, 0, 1, 1, 1, None, 1]
    out = smooth_possession(raw, min_hold=1)
    assert out == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]   # None inherits current holder


def test_smooth_possession_hysteresis_ignores_single_frame_blip():
    # Team 0 holds; a single frame where team 1 is nearest should NOT flip it.
    raw = [0, 0, 0, 1, 0, 0, 0]
    out = smooth_possession(raw, min_hold=3)
    assert out == [0, 0, 0, 0, 0, 0, 0]


def test_smooth_possession_flips_after_min_hold():
    raw = [0, 0, 1, 1, 1, 1]
    out = smooth_possession(raw, min_hold=3)
    # Switches to team 1 only once it has held for 3 consecutive frames.
    assert out == [0, 0, 0, 0, 1, 1]


# ---------------------------------------------------------------------------
# Full possession over a sequence
# ---------------------------------------------------------------------------
def test_compute_possession_percentages():
    roles = {1: ("player", 0), 2: ("player", 1)}
    frames = []
    # 6 frames team 0 has the ball, 4 frames team 1 has it.
    for f in range(1, 7):
        frames.append(_frame(f, [_track(99, "ball", 500, 400, 8, 8),
                                  _track(1, "player", 505, 380),
                                  _track(2, "player", 900, 380)]))
    for f in range(7, 11):
        frames.append(_frame(f, [_track(99, "ball", 900, 400, 8, 8),
                                  _track(1, "player", 505, 380),
                                  _track(2, "player", 905, 380)]))
    result = compute_possession(frames, roles, "ball", gate_factor=3.0, min_hold=1)
    assert result.counts == {0: 6, 1: 4}
    assert result.percentages == {0: 60.0, 1: 40.0}
    assert result.total_frames == 10
    # Cumulative ends at the final split.
    assert result.cumulative[-1][1:] == (60.0, 40.0)


def test_export_possession_roundtrip(tmp_path):
    roles = {1: ("player", 0)}
    frames = [_frame(f, [_track(99, "ball", 500, 400, 8, 8),
                         _track(1, "player", 505, 380)]) for f in range(1, 5)]
    result = compute_possession(frames, roles, "ball", gate_factor=3.0, min_hold=1)
    path = export_possession(result, tmp_path / "p.json", metadata={"x": 1})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["possession_percent"]["0"] == 100.0
    assert len(data["timeline"]) == 4
