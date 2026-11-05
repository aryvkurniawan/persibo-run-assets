"""Phase 5 — write the speed/distance analytics JSON.

Schema (``outputs/<video>_analytics.json``)::

    {
      "metadata": {"phase": "speed_distance", "fps": 25, ...},
      "tracks":   [ {track_id, role, team_id, total_distance_m,
                     avg_speed_kmh, max_speed_kmh, valid_samples,
                     invalid_jumps, insufficient_data}, ... ],
      "summary":  {"team_0_total_distance_m": ...,
                   "team_1_total_distance_m": ...,
                   "total_distance_m": ...,
                   "distance_by_role": {...}}
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from analytics.analytics_models import AnalyticsResult
from utils.logger import get_logger

logger = get_logger("analytics.analytics_exporter")


def _player_label(obj) -> str:
    name = getattr(obj, "player_name", None)
    display = getattr(obj, "player_display_name", None) or f"Player {obj.track_id}"
    return f"{name} (#{obj.track_id})" if name else display


def build_summary(result: AnalyticsResult) -> dict:
    """Team / role distance totals in the exported ``summary`` shape."""
    summary: dict = {}
    for team_id, dist in sorted(
            result.distance_by_team.items(), key=lambda kv: (kv[0] is None, kv[0])):
        key = (f"team_{team_id}_total_distance_m" if team_id is not None
               else "no_team_total_distance_m")
        summary[key] = round(float(dist), 2)
    summary["total_distance_m"] = round(float(result.total_distance_m), 2)
    summary["distance_by_role"] = {
        role: round(float(dist), 2)
        for role, dist in sorted(result.distance_by_role.items())
    }
    return summary


def export_analytics(
    result: AnalyticsResult,
    path: str | Path,
    metadata: Optional[dict] = None,
) -> Path:
    """Write the analytics JSON and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {"phase": "speed_distance", "fps": result.fps,
                     **(metadata or {})},
        "tracks": [t.to_dict() for t in result.tracks],
        "summary": build_summary(result),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Wrote analytics for %d tracks to %s", len(result.tracks), path)
    return path


def export_analytics_txt(
    result: AnalyticsResult,
    path: str | Path,
    title: str = "",
) -> Path:
    """Write a human-readable .txt report (team totals + a per-track table)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    head = f"SPEED & DISTANCE REPORT{' — ' + title if title else ''}"
    lines.append(head)
    lines.append("=" * max(52, len(head)))
    lines.append(f"fps: {result.fps:g}    tracks: {len(result.tracks)}")
    lines.append("")
    lines.append("TEAM TOTALS")
    for team_id in sorted(result.distance_by_team,
                          key=lambda t: (t is None, t)):
        label = f"Team {team_id}" if team_id is not None else "No team"
        lines.append(f"  {label:<10}: {result.distance_by_team[team_id]:8.1f} m")
    lines.append(f"  {'TOTAL':<10}: {result.total_distance_m:8.1f} m")
    if result.distance_by_role:
        lines.append("")
        lines.append("BY ROLE")
        for role, dist in sorted(result.distance_by_role.items()):
            lines.append(f"  {role:<12}: {dist:8.1f} m")
    lines.append("")
    lines.append(f"{'PLAYER':<24}{'ROLE':<11}{'TEAM':<6}"
                 f"{'DIST(m)':>9}{'AVG(km/h)':>11}{'MAX(km/h)':>11}{'  NOTE'}")
    lines.append("-" * 80)
    for t in sorted(result.tracks, key=lambda t: -t.total_distance_m):
        team = "-" if t.team_id is None else str(t.team_id)
        note = "low-data" if t.insufficient_data else ""
        lines.append(
            f"{_player_label(t):<24}{t.role:<11}{team:<6}"
            f"{t.total_distance_m:9.1f}{t.avg_speed_kmh:11.1f}"
            f"{t.max_speed_kmh:11.1f}  {note}")
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote analytics report to %s", path)
    return path


def export_player_analytics(
    players: Sequence,
    path: str | Path,
    metadata: Optional[dict] = None,
) -> Path:
    """Write the Phase 6 player-performance JSON (one row per player).

    ``players`` is a sequence of ``PlayerPerformance`` (each with ``to_dict``).
    Output: ``outputs/<video>_player_analytics.json``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {"phase": "player_performance", **(metadata or {})},
        "players": [p.to_dict() for p in players],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Wrote player analytics for %d players to %s", len(players), path)
    return path


def export_player_analytics_txt(
    players: Sequence,
    path: str | Path,
    title: str = "",
) -> Path:
    """Write a human-readable player-performance report (ranked by rating)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = f"PLAYER PERFORMANCE REPORT{' — ' + title if title else ''}"
    lines = [head, "=" * max(56, len(head)), f"players rated: {len(players)}", ""]
    ranked = sorted(players, key=lambda p: -p.player_rating)
    for rank, p in enumerate(ranked, start=1):
        team = "-" if p.team_id is None else str(p.team_id)
        lines.append(
            f"{rank:>2}. {_player_label(p):<22} T{team:<3} {p.role:<10} "
            f"RATING {p.player_rating:>4}/10  ({p.performance_status})")
        lines.append(
            f"      work rate: {p.work_rate:<6} | activity: {p.activity_level:<6} "
            f"| fatigue: {p.fatigue_level:<6}")
        lines.append(
            f"      distance: {p.total_distance_m:6.1f} m | "
            f"avg {p.avg_speed_kmh:.1f} km/h | max {p.max_speed_kmh:.1f} km/h")
        lines.append(f"      > {p.insight}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote player report for %d players to %s", len(players), path)
    return path
