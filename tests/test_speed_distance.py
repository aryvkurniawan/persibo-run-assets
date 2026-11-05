"""Tests for Phase 5 — speed & distance analytics."""

from __future__ import annotations

import json

import pytest

from analytics.analytics_exporter import (
    build_summary,
    export_analytics,
    export_analytics_txt,
)
from analytics.analytics_models import AnalyticsResult, TrackAnalytics
from analytics.analytics_visualizer import (
    build_speed_lookup,
    build_stat_lookup,
    draw_player_stat_bars,
    draw_speed_labels,
)
from analytics.speed_distance import (
    compute_analytics,
    compute_track_analytics,
    moving_average,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _frame(idx, objects):
    return {"frame": idx, "objects": objects}


def _obj(tid, x, y, role="player", team=0, on_pitch=True):
    return {"track_id": tid, "role": role, "team_id": team,
            "field": [x, y], "on_pitch": on_pitch}


# ---------------------------------------------------------------------------
# Speed calculation
# ---------------------------------------------------------------------------
def test_speed_from_constant_velocity():
    # Moves 1 m every frame at 10 fps -> 10 m/s -> 36 km/h.
    samples = [(i, float(i), 0.0) for i in range(1, 11)]
    ta = compute_track_analytics(7, "player", 0, samples, fps=10,
                                 smoothing_window=1, max_speed_kmh=50)
    assert ta.total_distance_m == pytest.approx(9.0)        # 9 one-metre steps
    assert ta.avg_speed_kmh == pytest.approx(36.0, abs=1e-6)
    assert ta.max_speed_kmh == pytest.approx(36.0, abs=1e-6)
    assert ta.valid_samples == 9
    assert ta.invalid_jumps == 0


def test_speed_uses_frame_gap_for_dt():
    # A 2-metre move across a 2-frame gap at 10 fps = 1 m / 0.1 s steps... no:
    # 2 m over 0.2 s = 10 m/s = 36 km/h.
    samples = [(1, 0.0, 0.0), (3, 2.0, 0.0)]
    ta = compute_track_analytics(1, "player", 0, samples, fps=10,
                                 smoothing_window=1, max_speed_kmh=50, min_samples=1)
    assert ta.avg_speed_kmh == pytest.approx(36.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------
def test_distance_sums_valid_steps_only():
    # Three 1 m steps, but the 2nd is an impossible 30 m jump -> excluded.
    samples = [(1, 0.0, 0.0), (2, 1.0, 0.0), (3, 31.0, 0.0), (4, 32.0, 0.0)]
    ta = compute_track_analytics(1, "player", 0, samples, fps=10,
                                 smoothing_window=1, max_speed_kmh=38, min_samples=1)
    # Steps: 1m (ok), 30m (jump, rejected), 1m (ok) -> 2 m total.
    assert ta.total_distance_m == pytest.approx(2.0)
    assert ta.invalid_jumps == 1
    assert ta.valid_samples == 2


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def test_moving_average_smooths_spike():
    out = moving_average([10, 10, 30, 10, 10], window=3)
    assert out[2] < 30                       # the spike is pulled down
    assert len(out) == 5


def test_moving_average_noop_for_window_one():
    assert moving_average([1, 2, 3], window=1) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Invalid-jump filtering
# ---------------------------------------------------------------------------
def test_invalid_jump_not_counted_in_distance_or_speed():
    # A single huge teleport (homography blow-up) between two points.
    samples = [(1, 0.0, 0.0), (2, 500.0, 500.0)]
    ta = compute_track_analytics(1, "player", 0, samples, fps=25,
                                 smoothing_window=1, max_speed_kmh=38, min_samples=1)
    assert ta.total_distance_m == 0.0
    assert ta.invalid_jumps == 1
    assert ta.valid_samples == 0
    assert ta.max_speed_kmh == 0.0


def test_insufficient_data_flag():
    samples = [(1, 0.0, 0.0), (2, 1.0, 0.0)]      # 1 valid sample
    ta = compute_track_analytics(1, "player", 0, samples, fps=10,
                                 smoothing_window=1, max_speed_kmh=50, min_samples=5)
    assert ta.insufficient_data is True


# ---------------------------------------------------------------------------
# Aggregation (team / role) and off-pitch / ball handling
# ---------------------------------------------------------------------------
def _moving_match():
    frames = []
    for i in range(1, 6):
        frames.append(_frame(i, [
            _obj(1, float(i), 0.0, team=0),            # team 0 player moves
            _obj(2, float(i) * 0.5, 5.0, team=1),      # team 1 player (slower)
            _obj(9, 52.5, 34.0, role="ball", team=None),   # ball ignored
        ]))
    return frames


def test_team_summary_split():
    frames = _moving_match()
    roles = {1: ("player", 0), 2: ("player", 1), 9: ("ball", None)}
    result = compute_analytics(frames, roles, fps=10, smoothing_window=1, min_track_samples=1)
    # Two player tracks, ball excluded.
    assert {t.track_id for t in result.tracks} == {1, 2}
    d0 = result.team_distance(0)
    d1 = result.team_distance(1)
    assert d0 == pytest.approx(4.0)            # 4 one-metre steps
    assert d1 == pytest.approx(2.0)            # 4 half-metre steps
    assert result.total_distance_m == pytest.approx(6.0)


def test_referees_and_unknowns_are_excluded():
    frames = []
    for i in range(1, 6):
        frames.append(_frame(i, [
            _obj(1, float(i), 0.0, role="player", team=0),
            _obj(8, float(i), 10.0, role="referee", team=None),
            _obj(9, float(i), 20.0, role="unknown", team=None),
        ]))
    roles = {1: ("player", 0), 8: ("referee", None), 9: ("unknown", None)}
    result = compute_analytics(frames, roles, fps=10, smoothing_window=1, min_track_samples=1)
    assert {t.track_id for t in result.tracks} == {1}     # only the player
    assert "referee" not in result.distance_by_role
    assert "unknown" not in result.distance_by_role


def test_off_pitch_positions_are_ignored():
    frames = [
        _frame(1, [_obj(1, 0.0, 0.0)]),
        _frame(2, [_obj(1, 999.0, 999.0, on_pitch=False)]),   # blow-up, dropped
        _frame(3, [_obj(1, 1.0, 0.0)]),
    ]
    roles = {1: ("player", 0)}
    result = compute_analytics(frames, roles, fps=10, smoothing_window=1,
                               min_samples=1, min_track_samples=1)
    track = result.tracks[0]
    # Only frames 1 and 3 used -> one 1 m step across a 2-frame gap.
    assert track.total_distance_m == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Export + summary + speed lookup
# ---------------------------------------------------------------------------
def test_export_analytics_json(tmp_path):
    frames = _moving_match()
    roles = {1: ("player", 0), 2: ("player", 1)}
    result = compute_analytics(frames, roles, fps=25, smoothing_window=1,
                               max_speed_kmh=200, min_track_samples=1)
    path = export_analytics(result, tmp_path / "x_analytics.json",
                            metadata={"source": "unit"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metadata"]["phase"] == "speed_distance"
    assert data["metadata"]["fps"] == 25
    t = {row["track_id"]: row for row in data["tracks"]}
    assert set(t) == {1, 2}
    assert t[1]["role"] == "player" and t[1]["team_id"] == 0
    assert "team_0_total_distance_m" in data["summary"]
    assert "team_1_total_distance_m" in data["summary"]
    assert data["summary"]["total_distance_m"] == pytest.approx(
        result.total_distance_m, abs=0.01)


def test_export_analytics_txt_report(tmp_path):
    frames = _moving_match()
    roles = {1: ("player", 0), 2: ("player", 1)}
    result = compute_analytics(frames, roles, fps=25, smoothing_window=1,
                               max_speed_kmh=200, min_track_samples=1)
    path = export_analytics_txt(result, tmp_path / "x_analytics.txt", title="x")
    text = path.read_text(encoding="utf-8")
    assert "SPEED & DISTANCE REPORT" in text
    assert "TEAM TOTALS" in text
    assert "Team 0" in text and "Team 1" in text
    assert "Player 1" in text and "Player 2" in text


def test_player_names_propagate_to_speed_json_and_report(tmp_path):
    frames = _moving_match()
    roles = {1: ("player", 0), 2: ("player", 1)}
    result = compute_analytics(
        frames, roles, fps=25, smoothing_window=1, max_speed_kmh=200,
        min_track_samples=1, player_names={1: "Ada Hegerberg"})

    path = export_analytics(result, tmp_path / "x_analytics.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {r["track_id"]: r for r in data["tracks"]}[1]
    assert row["player_name"] == "Ada Hegerberg"
    assert row["player_display_name"] == "Ada Hegerberg"

    txt = export_analytics_txt(result, tmp_path / "x_analytics.txt")
    assert "Ada Hegerberg (#1)" in txt.read_text(encoding="utf-8")


def test_empty_player_name_exports_stable_display_name(tmp_path):
    frames = _moving_match()
    roles = {1: ("player", 0), 2: ("player", 1)}
    result = compute_analytics(
        frames, roles, fps=25, smoothing_window=1, max_speed_kmh=200,
        min_track_samples=1)

    path = export_analytics(result, tmp_path / "x_analytics.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {r["track_id"]: r for r in data["tracks"]}[1]
    assert row["player_display_name"] == "Player 1"
    assert "player_name" not in row


def test_build_speed_lookup_maps_frame_track_to_kmh():
    track = TrackAnalytics(7, "player", 0)
    track.frame_speed_kmh = {10: 18.4, 11: 19.0}
    result = AnalyticsResult(fps=25, tracks=[track])
    lookup = build_speed_lookup(result)
    assert lookup[(10, 7)] == pytest.approx(18.4)
    assert lookup[(11, 7)] == pytest.approx(19.0)


def test_running_distance_is_cumulative():
    samples = [(i, float(i), 0.0) for i in range(1, 6)]   # 1 m/frame
    ta = compute_track_analytics(7, "player", 0, samples, fps=10,
                                 smoothing_window=1, max_speed_kmh=50, min_samples=1)
    # Cumulative distance grows 1, 2, 3, 4 m over frames 2..5.
    assert ta.frame_distance_m[2] == pytest.approx(1.0)
    assert ta.frame_distance_m[5] == pytest.approx(4.0)


def test_build_stat_lookup_pairs_speed_and_distance():
    track = TrackAnalytics(7, "player", 0)
    track.frame_speed_kmh = {10: 18.4}
    track.frame_distance_m = {10: 92.0}
    result = AnalyticsResult(fps=25, tracks=[track])
    lookup = build_stat_lookup(result)
    speed, dist = lookup[(10, 7)]
    assert speed == pytest.approx(18.4) and dist == pytest.approx(92.0)


def test_draw_player_stat_bars_changes_frame():
    import numpy as np

    class _Track:
        def __init__(self, tid, bbox):
            self.track_id, self.class_name, self.bbox = tid, "player", bbox

    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    tracks = [_Track(7, (180, 150, 220, 250))]
    draw_player_stat_bars(frame, tracks, {(5, 7): (18.4, 92.0)}, frame_index=5)
    assert np.any(frame != 0)                       # a panel was drawn


def test_draw_player_stat_bars_uses_player_name(monkeypatch):
    import cv2
    import numpy as np

    class _Track:
        def __init__(self, tid, bbox):
            self.track_id, self.class_name, self.bbox = tid, "player", bbox

    drawn = []

    def fake_put_text(img, text, *args, **kwargs):
        drawn.append(text)
        return img

    monkeypatch.setattr(cv2, "putText", fake_put_text)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    tracks = [_Track(7, (180, 150, 220, 250))]
    draw_player_stat_bars(
        frame, tracks, {(5, 7): (18.4, 92.0)}, frame_index=5,
        names_map={7: "Marta"})

    assert "Marta" in drawn


def test_draw_speed_labels_uses_player_name_or_fallback(monkeypatch):
    import cv2
    import numpy as np

    class _Track:
        def __init__(self, tid, bbox):
            self.track_id, self.class_name, self.bbox = tid, "player", bbox

    drawn = []

    def fake_put_text(img, text, *args, **kwargs):
        drawn.append(text)
        return img

    monkeypatch.setattr(cv2, "putText", fake_put_text)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    tracks = [
        _Track(7, (180, 150, 220, 250)),
        _Track(8, (240, 150, 280, 250)),
    ]
    draw_speed_labels(
        frame, tracks, {(5, 7): 18.4, (5, 8): 16.2}, frame_index=5,
        names_map={7: "Marta"})

    assert "Marta | 18.4 km/h" in drawn
    assert "Player 8 | 16.2 km/h" in drawn


def test_build_summary_keys():
    result = AnalyticsResult(fps=25)
    result.distance_by_team = {0: 100.0, 1: 80.0, None: 12.0}
    result.distance_by_role = {"player": 180.0, "referee": 12.0}
    result.total_distance_m = 192.0
    summary = build_summary(result)
    assert summary["team_0_total_distance_m"] == 100.0
    assert summary["team_1_total_distance_m"] == 80.0
    assert summary["no_team_total_distance_m"] == 12.0
    assert summary["distance_by_role"]["player"] == 180.0


# ---------------------------------------------------------------------------
# Safety: empty / missing input
# ---------------------------------------------------------------------------
def test_empty_frames_yield_empty_result():
    result = compute_analytics([], {}, fps=25)
    assert result.tracks == []
    assert result.total_distance_m == 0.0
