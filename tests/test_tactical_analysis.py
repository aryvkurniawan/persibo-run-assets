"""Tests for Phase 8 — team tactical analysis."""

from __future__ import annotations

import json

from analytics.analytics_models import AnalyticsResult, TrackAnalytics
from analytics.tactical_analysis import (
    attacking_zone,
    classify_buildup,
    classify_compactness,
    classify_pressure,
    classify_shape,
    classify_transition,
    compute_tactical_analysis,
    compute_team_tactics,
    export_team_tactical,
    export_team_tactical_txt,
    spatial_stats,
)


def _frame(idx, objects):
    return {"frame": idx, "objects": objects}


def _obj(tid, x, y, team=0, role="player", on_pitch=True):
    return {"track_id": tid, "role": role, "team_id": team,
            "field": [x, y], "on_pitch": on_pitch}


# ---------------------------------------------------------------------------
# Compactness
# ---------------------------------------------------------------------------
def test_classify_compactness_bands():
    assert classify_compactness(10.0)[0] == "Compact"
    assert classify_compactness(22.0)[0] == "Balanced"
    assert classify_compactness(30.0)[0] == "Wide"


def test_spatial_stats_compactness_reflects_spread():
    tight = [[(50, 34), (51, 34), (49, 34), (50, 35)]]
    spread = [[(20, 10), (90, 60), (50, 34), (80, 20)]]
    assert spatial_stats(tight)["compactness_m"] < spatial_stats(spread)["compactness_m"]


# ---------------------------------------------------------------------------
# Width / shape
# ---------------------------------------------------------------------------
def test_classify_shape_bands():
    assert classify_shape(8.0)[0] == "Compact"
    assert classify_shape(15.0)[0] == "Balanced"
    assert classify_shape(22.0)[0] == "Wide"


def test_width_increases_with_lateral_spread():
    narrow = [[(50, 33), (50, 34), (50, 35)]]
    wide = [[(50, 5), (50, 34), (50, 63)]]
    assert spatial_stats(wide)["width_m"] > spatial_stats(narrow)["width_m"]


# ---------------------------------------------------------------------------
# Attacking zones
# ---------------------------------------------------------------------------
def test_attacking_zone_picks_dominant_third():
    assert attacking_zone((0.7, 0.2, 0.1))[0] == "Left"
    assert attacking_zone((0.1, 0.8, 0.1))[0] == "Center"
    assert attacking_zone((0.1, 0.2, 0.7))[0] == "Right"


def test_spatial_stats_lateral_fractions_sum_to_one():
    pts = [[(50, 5), (50, 34), (50, 60), (50, 10)]]
    fr = spatial_stats(pts)["lateral_fractions"]
    assert abs(sum(fr) - 1.0) < 1e-6
    assert fr[0] > fr[2]                        # more on the left (low y)


# ---------------------------------------------------------------------------
# Transition / build-up / pressure
# ---------------------------------------------------------------------------
def test_classify_transition_bands():
    assert classify_transition(11.0)[0] == "Fast"
    assert classify_transition(8.0)[0] == "Medium"
    assert classify_transition(5.0)[0] == "Slow"


def test_buildup_direct_when_stretched_or_fast():
    assert classify_buildup(25.0, "Medium")[0] == "Direct Style"
    assert classify_buildup(12.0, "Fast")[0] == "Direct Style"
    assert classify_buildup(12.0, "Slow")[0] == "Short Passing Style"


def test_pressure_bands():
    assert classify_pressure(65.0)[0] == "High Press"
    assert classify_pressure(50.0)[0] == "Mid Block"
    assert classify_pressure(30.0)[0] == "Low Block"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def _match_frames():
    frames = []
    for i in range(1, 11):
        frames.append(_frame(i, [
            # team 0 sits deep on the left (low x), team 1 high on the right.
            _obj(1, 20.0, 20.0, team=0), _obj(2, 25.0, 40.0, team=0),
            _obj(3, 30.0, 30.0, team=0),
            _obj(4, 80.0, 30.0, team=1), _obj(5, 85.0, 35.0, team=1),
            _obj(6, 78.0, 25.0, team=1),
        ]))
    return frames


def test_compute_tactical_analysis_two_teams_with_reasons():
    result = AnalyticsResult(fps=25, tracks=[
        TrackAnalytics(1, "player", 0, avg_speed_kmh=8.0, valid_samples=100),
        TrackAnalytics(4, "player", 1, avg_speed_kmh=10.0, valid_samples=100),
    ])
    profiles = compute_tactical_analysis(_match_frames(), result)
    assert {p.team_id for p in profiles} == {0, 1}
    for p in profiles:
        assert p.compactness in {"Compact", "Balanced", "Wide"}
        assert p.pressure_style in {"High Press", "Mid Block", "Low Block"}
        assert p.attacking_zone in {"Left", "Center", "Right"}
        assert p.reasons["compactness"]            # non-empty explanation
    # team 1 is higher up the pitch -> attacks right (own goal at 0) and its
    # block sits further from its own goal than team 0's.
    by = {p.team_id: p for p in profiles}
    assert by[1].metrics["block_height_m"] > by[0].metrics["block_height_m"]


def test_export_team_tactical_json(tmp_path):
    profiles = compute_tactical_analysis(_match_frames())
    path = export_team_tactical(profiles, tmp_path / "v_team_tactical.json",
                                metadata={"source": "unit"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metadata"]["phase"] == "team_tactical"
    row = data["teams"][0]
    assert set(row) >= {"team_id", "compactness", "team_shape", "attacking_zone",
                        "transition_speed", "build_up_style", "pressure_style",
                        "metrics", "reasons"}


def test_export_team_tactical_txt(tmp_path):
    profiles = compute_tactical_analysis(_match_frames())
    path = export_team_tactical_txt(profiles, tmp_path / "v_tactical.txt", title="v")
    text = path.read_text(encoding="utf-8")
    assert "TEAM TACTICAL ANALYSIS" in text
    assert "TEAM 0" in text and "TEAM 1" in text
    assert "Pressure Style" in text and "Compactness" in text
