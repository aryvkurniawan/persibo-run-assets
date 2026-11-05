"""Tests for Phase 10 — AI match intelligence (final report)."""

from __future__ import annotations

import json

from analytics.match_report import (
    build_match_report,
    export_final_report,
    export_final_report_txt,
    impact_from_rating,
    team_style,
)
from analytics.player_analytics import PlayerPerformance
from analytics.possession_engine import PossessionAnalysis
from analytics.tactical_analysis import TacticalProfile


def _player(tid, team, rating, fatigue="Low", insight="Active.", name=None):
    return PlayerPerformance(
        track_id=tid, team_id=team, role="player",
        total_distance_m=80.0 + rating, avg_speed_kmh=10.0, max_speed_kmh=28.0,
        work_rate="High", activity_level="High", fatigue_level=fatigue,
        performance_status="Good", player_rating=rating, insight=insight,
        player_name=name)


def _profile(team, pressure="High Press", build="Direct Style",
             compact="Compact", transition="Fast"):
    return TacticalProfile(
        team_id=team, compactness=compact, team_shape="Balanced",
        attacking_zone="Center", transition_speed=transition,
        build_up_style=build, pressure_style=pressure)


def _possession(p0, p1, dominant):
    possessed = 100
    return PossessionAnalysis(
        percentages={0: p0, 1: p1}, counts={0: int(p0), 1: int(p1)},
        possessed_frames=possessed, total_frames=possessed, dominant_team=dominant)


# ---------------------------------------------------------------------------
# Mappers
# ---------------------------------------------------------------------------
def test_impact_from_rating():
    assert impact_from_rating(8.5) == "High"
    assert impact_from_rating(6.0) == "Medium"
    assert impact_from_rating(3.0) == "Low"


def test_team_style_mapping():
    assert team_style(_profile(0, pressure="High Press")) == "Attacking"
    assert team_style(_profile(0, pressure="Mid Block",
                               build="Short Passing Style")) == "Possession"
    assert team_style(_profile(0, pressure="Mid Block",
                               build="Direct Style")) == "Counter-Attacking"
    assert team_style(_profile(0, pressure="Low Block",
                               build="Mixed")) == "Defensive"


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------
def test_man_of_the_match_and_weakest():
    players = [_player(1, 0, 9.0), _player(2, 1, 4.0), _player(3, 0, 7.0)]
    report = build_match_report(players, [_profile(0), _profile(1)],
                                _possession(60.0, 40.0, 0))
    assert report.man_of_the_match == 1                # highest rating
    assert report.weakest_player == 2                  # lowest rating
    assert report.dominant_team == "Team 0"


def test_dominant_team_falls_back_to_distance_without_possession():
    players = [_player(1, 0, 8.0), _player(2, 1, 6.0)]   # team 0 covers more
    report = build_match_report(players, [_profile(0), _profile(1)], None)
    assert report.dominant_team == "Team 0"


def test_player_report_impact_and_fields():
    players = [_player(7, 0, 8.7, fatigue="Medium", insight="Very active.")]
    report = build_match_report(players, [_profile(0)], _possession(55.0, 45.0, 0))
    p = report.players[0]
    assert p.to_dict() == {
        "track_id": 7, "team_id": 0, "player_display_name": "Player 7",
        "rating": 8.7,
        "fatigue": "Medium", "impact": "High", "insight": "Very active."}


def test_team_report_shape():
    report = build_match_report([_player(1, 0, 7.0)], [_profile(0)], None)
    t = report.teams[0].to_dict()
    assert set(t) == {"team_id", "style", "pressure", "compactness",
                      "transition_speed"}
    assert t["style"] == "Attacking"


def test_insights_and_recommendations_generated():
    players = [_player(1, 0, 9.0), _player(2, 1, 4.0)]
    report = build_match_report(players, [_profile(0), _profile(1)],
                                _possession(70.0, 30.0, 0))
    assert any("possession" in s.lower() for s in report.key_insights)
    assert report.recommendations                       # at least one rec
    # Team 1 had <45% possession -> a retention recommendation.
    assert any("retention" in r.lower() or "possession" in r.lower()
               for r in report.recommendations)


def test_summary_mentions_dominant_and_motm():
    players = [_player(7, 0, 9.0)]
    report = build_match_report(players, [_profile(0)], _possession(62.0, 38.0, 0))
    assert "Team 0" in report.summary
    assert "Player 7" in report.summary


def test_player_names_appear_in_match_report_outputs(tmp_path):
    players = [
        _player(7, 0, 9.0, name="Marta"),
        _player(8, 1, 5.0, name="Mia Hamm"),
    ]
    report = build_match_report(players, [_profile(0), _profile(1)], None)
    assert report.man_of_the_match == 7
    assert report.man_of_the_match_name == "Marta"
    assert report.man_of_the_match_display_name == "Marta"
    assert "Marta (#7)" in report.summary
    assert "Marta (#7)" in report.key_insights[0]

    jpath = export_final_report(report, tmp_path / "v_final_report.json")
    data = json.loads(jpath.read_text(encoding="utf-8"))
    assert data["man_of_the_match_name"] == "Marta"
    assert data["man_of_the_match_display_name"] == "Marta"
    assert data["players"][0]["player_name"] == "Marta"
    assert data["players"][0]["player_display_name"] == "Marta"

    tpath = export_final_report_txt(report, tmp_path / "v_final_report.txt")
    assert "Marta (#7)" in tpath.read_text(encoding="utf-8")


def test_empty_player_name_uses_match_report_display_name(tmp_path):
    players = [_player(12, 0, 8.0)]
    report = build_match_report(players, [_profile(0)], None)
    assert report.man_of_the_match_name is None
    assert report.man_of_the_match_display_name == "Player 12"
    assert "Player 12" in report.summary

    jpath = export_final_report(report, tmp_path / "v_final_report.json")
    data = json.loads(jpath.read_text(encoding="utf-8"))
    assert data["man_of_the_match_name"] is None
    assert data["man_of_the_match_display_name"] == "Player 12"
    assert data["players"][0]["player_display_name"] == "Player 12"
    assert "player_name" not in data["players"][0]


def test_empty_players_is_safe():
    report = build_match_report([], [], None)
    assert report.man_of_the_match is None
    assert report.summary                               # still a string


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def test_export_final_report_json_and_txt(tmp_path):
    players = [_player(1, 0, 9.0), _player(2, 1, 5.0)]
    report = build_match_report(players, [_profile(0), _profile(1)],
                                _possession(58.0, 42.0, 0))
    jpath = export_final_report(report, tmp_path / "v_final_report.json",
                                metadata={"source": "unit"})
    data = json.loads(jpath.read_text(encoding="utf-8"))
    assert data["metadata"]["phase"] == "match_report"
    assert set(data) >= {"summary", "dominant_team", "man_of_the_match",
                         "weakest_player", "teams", "players", "key_insights",
                         "recommendations"}

    tpath = export_final_report_txt(report, tmp_path / "v_final_report.txt",
                                    title="v")
    text = tpath.read_text(encoding="utf-8")
    assert "MATCH REPORT" in text
    assert "RECOMMENDATIONS" in text and "KEY INSIGHTS" in text
