"""Phase 10 — AI match intelligence layer (deterministic analyst report).

Combines the earlier analytics — player performance (Phase 6), team tactics
(Phase 8) and ball possession (Phase 9) — into a single explainable match
report: a summary, the dominant team, man of the match, weakest player,
per-team tactical capsule, key insights and recommendations. Every line is
derived from the computed metrics with plain rules — NO external LLM, fully
reproducible. Reads other phases' outputs only; modifies nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from analytics.player_analytics import PlayerPerformance
from analytics.possession_engine import PossessionAnalysis
from analytics.tactical_analysis import TacticalProfile
from utils.logger import get_logger

logger = get_logger("analytics.match_report")


# ---------------------------------------------------------------------------
# Small deterministic mappers
# ---------------------------------------------------------------------------
def impact_from_rating(rating: float) -> str:
    if rating >= 7.5:
        return "High"
    if rating >= 5.0:
        return "Medium"
    return "Low"


def team_style(profile: TacticalProfile) -> str:
    """One-word playing style from the tactical labels (deterministic)."""
    if profile.pressure_style == "High Press":
        return "Attacking"
    if profile.build_up_style == "Short Passing Style":
        return "Possession"
    if profile.build_up_style == "Direct Style":
        return "Counter-Attacking"
    if profile.pressure_style == "Low Block":
        return "Defensive"
    return "Balanced"


# ---------------------------------------------------------------------------
# Report pieces
# ---------------------------------------------------------------------------
@dataclass
class PlayerReport:
    track_id: int
    team_id: Optional[int]
    rating: float
    fatigue: str
    impact: str
    insight: str
    player_name: Optional[str] = None

    @property
    def player_display_name(self) -> str:
        return self.player_name or f"Player {self.track_id}"

    def to_dict(self) -> dict:
        data = {
            "track_id": int(self.track_id),
            "team_id": (None if self.team_id is None else int(self.team_id)),
            "player_display_name": self.player_display_name,
            "rating": round(float(self.rating), 1),
            "fatigue": self.fatigue,
            "impact": self.impact,
            "insight": self.insight,
        }
        if self.player_name:
            data["player_name"] = self.player_name
        return data


@dataclass
class TeamReport:
    team_id: int
    style: str
    pressure: str
    compactness: str
    transition_speed: str

    def to_dict(self) -> dict:
        return {
            "team_id": int(self.team_id),
            "style": self.style,
            "pressure": self.pressure,
            "compactness": self.compactness,
            "transition_speed": self.transition_speed,
        }


@dataclass
class MatchReport:
    summary: str
    dominant_team: Optional[str]
    man_of_the_match: Optional[int]
    weakest_player: Optional[int]
    man_of_the_match_name: Optional[str] = None
    weakest_player_name: Optional[str] = None
    man_of_the_match_display_name: Optional[str] = None
    weakest_player_display_name: Optional[str] = None
    teams: List[TeamReport] = field(default_factory=list)
    players: List[PlayerReport] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "dominant_team": self.dominant_team,
            "man_of_the_match": self.man_of_the_match,
            "weakest_player": self.weakest_player,
            "man_of_the_match_name": self.man_of_the_match_name,
            "weakest_player_name": self.weakest_player_name,
            "man_of_the_match_display_name": self.man_of_the_match_display_name,
            "weakest_player_display_name": self.weakest_player_display_name,
            "teams": [t.to_dict() for t in self.teams],
            "players": [p.to_dict() for p in self.players],
            "key_insights": list(self.key_insights),
            "recommendations": list(self.recommendations),
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def _dominant_team(
    possession: Optional[PossessionAnalysis],
    players: Sequence[PlayerPerformance],
) -> Optional[int]:
    """Dominant team by possession, falling back to total distance covered."""
    if possession is not None and possession.dominant_team is not None:
        return possession.dominant_team
    dist: Dict[int, float] = {0: 0.0, 1: 0.0}
    for p in players:
        if p.team_id in (0, 1):
            dist[p.team_id] += p.total_distance_m
    if dist[0] == dist[1] == 0.0:
        return None
    return 0 if dist[0] >= dist[1] else 1


def build_match_report(
    players: Sequence[PlayerPerformance],
    profiles: Sequence[TacticalProfile],
    possession: Optional[PossessionAnalysis] = None,
) -> MatchReport:
    """Assemble the full match report from the phase outputs."""
    ranked = sorted(players, key=lambda p: -p.player_rating)
    player_reports = [
        PlayerReport(p.track_id, p.team_id, p.player_rating, p.fatigue_level,
                     impact_from_rating(p.player_rating), p.insight,
                     p.player_name)
        for p in ranked
    ]
    motm = ranked[0].track_id if ranked else None
    weakest = ranked[-1].track_id if ranked else None
    motm_name = ranked[0].player_name if ranked else None
    weakest_name = ranked[-1].player_name if ranked else None
    motm_display = ranked[0].player_display_name if ranked else None
    weakest_display = ranked[-1].player_display_name if ranked else None
    team_reports = [
        TeamReport(pr.team_id, team_style(pr), pr.pressure_style,
                   pr.compactness, pr.transition_speed)
        for pr in sorted(profiles, key=lambda p: p.team_id)
    ]
    dom = _dominant_team(possession, players)

    insights = _key_insights(ranked, team_reports, possession, dom)
    recs = _recommendations(player_reports, team_reports, possession, dom)
    summary = _summary(ranked, team_reports, possession, dom)

    return MatchReport(
        summary=summary,
        dominant_team=(f"Team {dom}" if dom is not None else None),
        man_of_the_match=motm,
        weakest_player=weakest,
        man_of_the_match_name=motm_name,
        weakest_player_name=weakest_name,
        man_of_the_match_display_name=motm_display,
        weakest_player_display_name=weakest_display,
        teams=team_reports,
        players=player_reports,
        key_insights=insights,
        recommendations=recs,
    )


def _team_label(team_id: Optional[int]) -> str:
    return f"Team {team_id}" if team_id is not None else "Unknown team"


def _player_label(player) -> str:
    name = getattr(player, "player_name", None)
    display = getattr(player, "player_display_name", None) or f"Player {player.track_id}"
    return f"{name} (#{player.track_id})" if name else display


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _key_insights(
    ranked: Sequence[PlayerPerformance],
    teams: Sequence[TeamReport],
    possession: Optional[PossessionAnalysis],
    dom: Optional[int],
) -> List[str]:
    out: List[str] = []
    if possession is not None and possession.possessed_frames:
        out.append(
            f"{_team_label(dom)} controlled possession "
            f"({possession.percentages.get(dom, 0.0):.0f}%).")
    if ranked:
        top = ranked[0]
        out.append(
            f"{_player_label(top)} ({_team_label(top.team_id)}) was the "
            f"standout at {top.player_rating}/10 — {top.insight}")
    for t in teams:
        out.append(
            f"{_team_label(t.team_id)} played {_article(t.style)} {t.style} game "
            f"with a {t.pressure} and {t.transition_speed.lower()} transitions.")
    tired = [p for p in ranked if p.fatigue_level == "High"]
    if len(tired) >= 3:
        out.append(f"{len(tired)} players showed high fatigue late on.")
    return out


def _recommendations(
    players: Sequence[PlayerReport],
    teams: Sequence[TeamReport],
    possession: Optional[PossessionAnalysis],
    dom: Optional[int],
) -> List[str]:
    out: List[str] = []
    if possession is not None and possession.possessed_frames and dom is not None:
        other = 1 - dom
        if possession.percentages.get(other, 0.0) < 45.0:
            out.append(
                f"{_team_label(other)} should improve ball retention and "
                "build-up to contest possession.")
    for t in teams:
        if t.pressure == "Low Block":
            out.append(
                f"{_team_label(t.team_id)} could push the defensive line "
                "higher to press and win the ball sooner.")
        if t.compactness == "Wide":
            out.append(
                f"{_team_label(t.team_id)} should stay more compact to deny "
                "central space.")
    for team_id in (0, 1):
        tired = [p for p in players
                 if p.team_id == team_id and p.fatigue == "High"]
        if len(tired) >= 3:
            out.append(
                f"{_team_label(team_id)} should manage workload / rotate — "
                "several players faded physically.")
    if not out:
        out.append("Both teams were well-matched; maintain current approach.")
    return out


def _summary(
    ranked: Sequence[PlayerPerformance],
    teams: Sequence[TeamReport],
    possession: Optional[PossessionAnalysis],
    dom: Optional[int],
) -> str:
    parts: List[str] = []
    if dom is not None:
        if possession is not None and possession.possessed_frames:
            parts.append(
                f"{_team_label(dom)} dominated the match with "
                f"{possession.percentages.get(dom, 0.0):.0f}% possession.")
        else:
            parts.append(f"{_team_label(dom)} was the more dominant side.")
    if ranked:
        top = ranked[0]
        parts.append(
            f"The standout performer was {_player_label(top)} "
            f"({_team_label(top.team_id)}), rated {top.player_rating}/10.")
    styles = ", ".join(f"{_team_label(t.team_id)} {t.style.lower()}"
                       for t in teams)
    if styles:
        parts.append(f"Tactically: {styles}.")
    return " ".join(parts) if parts else "Not enough data for a match summary."


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_final_report(
    report: MatchReport, path: str | Path, metadata: Optional[dict] = None
) -> Path:
    """Write the final report JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": {"phase": "match_report", **(metadata or {})},
               **report.to_dict()}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Wrote final match report to %s", path)
    return path


def export_final_report_txt(
    report: MatchReport, path: str | Path, title: str = ""
) -> Path:
    """Write a human-readable final match report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = f"MATCH REPORT{' — ' + title if title else ''}"
    lines = [head, "=" * max(56, len(head)), "", report.summary, ""]
    lines.append(f"Dominant team   : {report.dominant_team or 'n/a'}")
    if report.man_of_the_match is None:
        motm = "n/a"
    elif report.man_of_the_match_name:
        motm = f"{report.man_of_the_match_name} (#{report.man_of_the_match})"
    else:
        motm = (
            report.man_of_the_match_display_name
            or f"Player {report.man_of_the_match}"
        )
    if report.weakest_player is None:
        weakest = "n/a"
    elif report.weakest_player_name:
        weakest = f"{report.weakest_player_name} (#{report.weakest_player})"
    else:
        weakest = (
            report.weakest_player_display_name
            or f"Player {report.weakest_player}"
        )
    lines.append(f"Man of the match: {motm}")
    lines.append(f"Weakest player  : {weakest}")
    lines.append("")
    lines.append("TEAMS")
    for t in report.teams:
        lines.append(f"  Team {t.team_id}: {t.style} | {t.pressure} | "
                     f"{t.compactness} | {t.transition_speed} transitions")
    lines.append("")
    lines.append("KEY INSIGHTS")
    for s in report.key_insights:
        lines.append(f"  - {s}")
    lines.append("")
    lines.append("RECOMMENDATIONS")
    for s in report.recommendations:
        lines.append(f"  - {s}")
    lines.append("")
    lines.append("TOP PLAYERS")
    for p in report.players[:5]:
        label = (
            f"{p.player_name} (#{p.track_id})"
            if p.player_name else p.player_display_name)
        lines.append(f"  {label:<18} T{p.team_id}  rating {p.rating:>4}/10  "
                     f"impact {p.impact:<6} fatigue {p.fatigue:<6} | {p.insight}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote final match report (txt) to %s", path)
    return path
