"""Tests for Phase 9 — field-space ball possession."""

from __future__ import annotations

import json

from analytics.possession_engine import (
    compute_field_possession,
    export_field_possession,
    frame_owner,
)


def _frame(idx, objects):
    return {"frame": idx, "objects": objects}


def _obj(tid, x, y, team=0, role="player", on_pitch=True):
    return {"track_id": tid, "role": role, "team_id": team,
            "field": [x, y], "on_pitch": on_pitch}


def _ball(x, y):
    return {"track_id": 9, "role": "ball", "team_id": None,
            "field": [x, y], "on_pitch": True}


# ---------------------------------------------------------------------------
# Nearest-player ownership
# ---------------------------------------------------------------------------
def test_nearest_player_owns_the_ball():
    objs = [
        _ball(50.0, 34.0),
        _obj(1, 51.0, 34.0, team=0),     # 1 m from ball -> nearest
        _obj(2, 60.0, 34.0, team=1),     # 10 m away
    ]
    assert frame_owner(objs) == 0


def test_owner_respects_distance_gate():
    objs = [_ball(50.0, 34.0), _obj(1, 60.0, 34.0, team=1)]  # 10 m away
    assert frame_owner(objs, gate_m=5.0) is None             # loose ball
    assert frame_owner(objs, gate_m=20.0) == 1


def test_owner_skips_offpitch_and_non_team():
    objs = [
        _ball(50.0, 34.0),
        _obj(1, 50.5, 34.0, team=0, on_pitch=False),   # closest but off-pitch
        _obj(2, 55.0, 34.0, team=1),
        _obj(3, 50.2, 34.0, role="referee", team=None),  # not a team -> ignored
    ]
    assert frame_owner(objs) == 1


# ---------------------------------------------------------------------------
# Missing ball handling
# ---------------------------------------------------------------------------
def test_no_ball_means_no_owner():
    objs = [_obj(1, 50.0, 34.0, team=0)]
    assert frame_owner(objs) is None


def test_offpitch_ball_means_no_owner():
    objs = [{"role": "ball", "team_id": None, "field": [float("nan"), 0.0]},
            _obj(1, 50.0, 34.0, team=0)]
    assert frame_owner(objs) is None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def test_possession_aggregation_and_dominant_team():
    frames = []
    # 6 frames team 0 holds, 4 frames team 1 holds, 1 frame no ball.
    for i in range(6):
        frames.append(_frame(i, [_ball(50, 34), _obj(1, 50.5, 34, team=0),
                                 _obj(2, 70, 34, team=1)]))
    for i in range(6, 10):
        frames.append(_frame(i, [_ball(50, 34), _obj(1, 70, 34, team=0),
                                 _obj(2, 50.5, 34, team=1)]))
    frames.append(_frame(10, [_obj(1, 50, 34, team=0)]))   # no ball

    result = compute_field_possession(frames)
    assert result.counts == {0: 6, 1: 4}
    assert result.percentages[0] == 60.0
    assert result.percentages[1] == 40.0
    assert result.possessed_frames == 10
    assert result.total_frames == 11
    assert result.dominant_team == 0
    # Timeline records every frame (None for the ball-less one).
    assert result.timeline[-1] == (10, None)


def test_empty_possession_has_no_dominant():
    result = compute_field_possession([_frame(0, [_obj(1, 50, 34, team=0)])])
    assert result.possessed_frames == 0
    assert result.dominant_team is None
    assert result.percentages == {0: 0.0, 1: 0.0}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def test_export_field_possession_json(tmp_path):
    frames = [_frame(i, [_ball(50, 34), _obj(1, 50.5, 34, team=0),
                         _obj(2, 70, 34, team=1)]) for i in range(5)]
    result = compute_field_possession(frames)
    path = export_field_possession(result, tmp_path / "v_possession.json",
                                   metadata={"source": "unit"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["team_0_possession"] == 100.0
    assert data["team_1_possession"] == 0.0
    assert data["dominant_team"] == 0
    assert len(data["timeline"]) == 5
