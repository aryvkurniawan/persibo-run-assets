"""Tests for Phase 6 — player performance analytics."""

from __future__ import annotations

import json

import pytest

from analytics.analytics_exporter import (
    export_player_analytics,
    export_player_analytics_txt,
)
from analytics.analytics_models import AnalyticsResult, TrackAnalytics
from analytics.player_analytics import (
    compute_player_analytics,
    compute_player_performance,
    consistency,
    speed_drop,
)
from analytics.rating_engine import (
    classify_fatigue,
    classify_work_rate,
    generate_insight,
    performance_status,
    player_rating,
)


def _track(tid, role="player", team=0, dist=80.0, avg=10.0, mx=28.0,
           samples=300, speeds=None, name=None):
    t = TrackAnalytics(tid, role, team, total_distance_m=dist,
                       avg_speed_kmh=avg, max_speed_kmh=mx,
                       valid_samples=samples, player_name=name)
    if speeds is not None:
        t.frame_speed_kmh = dict(speeds)
    return t


# ---------------------------------------------------------------------------
# Rating generation
# ---------------------------------------------------------------------------
def test_rating_scales_with_metrics_and_is_bounded():
    low = player_rating(0.0, 0.0, 0.0, 0.0)
    high = player_rating(1.0, 1.0, 1.0, 1.0)
    assert low == 0.0
    assert high == 10.0
    assert 0.0 <= player_rating(0.5, 0.5, 0.5, 0.5) <= 10.0


def test_rating_increases_with_distance():
    assert player_rating(0.9, 0.5, 0.5, 0.5) > player_rating(0.2, 0.5, 0.5, 0.5)


def test_performance_status_bands():
    assert performance_status(9.0) == "Excellent"
    assert performance_status(6.5) == "Good"
    assert performance_status(4.5) == "Average"
    assert performance_status(2.0) == "Poor"


# ---------------------------------------------------------------------------
# Work-rate classification
# ---------------------------------------------------------------------------
def test_work_rate_classification():
    assert classify_work_rate(1.0, 1.0) == "High"
    assert classify_work_rate(0.5, 0.5) == "Medium"
    assert classify_work_rate(0.0, 0.0) == "Low"


# ---------------------------------------------------------------------------
# Fatigue calculation
# ---------------------------------------------------------------------------
def test_fatigue_high_when_distance_and_speed_drop_high():
    assert classify_fatigue(1.0, 1.0, 1.0) == "High"
    assert classify_fatigue(0.0, 0.0, 0.0) == "Low"


def test_speed_drop_detects_slowing_down():
    # First half ~20 km/h, second half ~10 km/h -> ~0.5 drop.
    speeds = {f: (20.0 if f <= 10 else 10.0) for f in range(1, 21)}
    drop = speed_drop(_track(1, speeds=speeds))
    assert drop == pytest.approx(0.5, abs=0.05)


def test_speed_drop_zero_when_steady():
    speeds = {f: 15.0 for f in range(1, 21)}
    assert speed_drop(_track(1, speeds=speeds)) == 0.0


def test_consistency_is_active_fraction():
    # 6 of 10 frames above the active threshold (6 km/h).
    speeds = {f: (8.0 if f <= 6 else 2.0) for f in range(1, 11)}
    assert consistency(_track(1, speeds=speeds)) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Insight generation
# ---------------------------------------------------------------------------
def test_insight_for_active_player():
    msg = generate_insight("player", "High", "High", "Low", 8.5, 31.0)
    assert "active" in msg.lower()
    assert "intensity" in msg.lower() or "pace" in msg.lower()


def test_insight_for_low_contribution():
    msg = generate_insight("player", "Low", "Low", "Low", 3.0, 18.0)
    assert "low movement" in msg.lower()


def test_insight_goalkeeper_is_specific():
    msg = generate_insight("goalkeeper", "Low", "Low", "Low", 4.0, 12.0)
    assert "keeper" in msg.lower()


# ---------------------------------------------------------------------------
# End-to-end over an AnalyticsResult
# ---------------------------------------------------------------------------
def test_compute_player_analytics_ranks_and_normalizes():
    busy_speeds = {f: 14.0 for f in range(1, 40)}
    lazy_speeds = {f: 3.0 for f in range(1, 40)}
    result = AnalyticsResult(fps=25, tracks=[
        _track(1, dist=120.0, avg=12.0, samples=300, speeds=busy_speeds),
        _track(2, dist=30.0, avg=4.0, samples=300, speeds=lazy_speeds),
    ])
    players = compute_player_analytics(result)
    assert [p.track_id for p in players] == [1, 2]      # ranked by rating desc
    assert players[0].player_rating > players[1].player_rating
    assert players[0].work_rate == "High"
    assert players[1].activity_level == "Low"


def test_compute_player_analytics_skips_insufficient_data():
    t = _track(9, samples=2)
    t.insufficient_data = True
    result = AnalyticsResult(fps=25, tracks=[t])
    assert compute_player_analytics(result) == []


def test_export_player_analytics_json(tmp_path):
    result = AnalyticsResult(fps=25, tracks=[
        _track(1, dist=100.0, avg=11.0, speeds={f: 12.0 for f in range(1, 40)}),
    ])
    players = compute_player_analytics(result)
    path = export_player_analytics(players, tmp_path / "v_player_analytics.json",
                                   metadata={"source": "unit"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metadata"]["phase"] == "player_performance"
    row = data["players"][0]
    assert set(row) >= {"track_id", "team_id", "role", "work_rate",
                        "activity_level", "fatigue_level", "performance_status",
                        "player_rating", "insight"}


def test_export_player_analytics_txt(tmp_path):
    result = AnalyticsResult(fps=25, tracks=[
        _track(1, dist=100.0, avg=11.0, speeds={f: 12.0 for f in range(1, 40)}),
        _track(2, dist=40.0, avg=5.0, speeds={f: 4.0 for f in range(1, 40)}),
    ])
    players = compute_player_analytics(result)
    path = export_player_analytics_txt(players, tmp_path / "v_player.txt", title="v")
    text = path.read_text(encoding="utf-8")
    assert "PLAYER PERFORMANCE REPORT" in text
    assert "RATING" in text
    assert "Player 1" in text and "Player 2" in text
    assert ">" in text                              # the insight line marker


def test_player_names_propagate_to_performance_exports(tmp_path):
    result = AnalyticsResult(fps=25, tracks=[
        _track(
            10, dist=100.0, avg=11.0,
            speeds={f: 12.0 for f in range(1, 40)},
            name="Sam Kerr",
        ),
    ])
    players = compute_player_analytics(result)

    path = export_player_analytics(players, tmp_path / "v_player_analytics.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["players"][0]["player_name"] == "Sam Kerr"
    assert data["players"][0]["player_display_name"] == "Sam Kerr"

    txt = export_player_analytics_txt(players, tmp_path / "v_player.txt")
    assert "Sam Kerr (#10)" in txt.read_text(encoding="utf-8")


def test_empty_player_name_uses_display_name_in_performance_export(tmp_path):
    result = AnalyticsResult(fps=25, tracks=[
        _track(11, dist=100.0, avg=11.0,
               speeds={f: 12.0 for f in range(1, 40)}),
    ])
    players = compute_player_analytics(result)

    path = export_player_analytics(players, tmp_path / "v_player_analytics.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["players"][0]["player_display_name"] == "Player 11"
    assert "player_name" not in data["players"][0]
