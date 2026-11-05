"""Phase 6 — player performance analytics.

Turns the Phase 5 :class:`~analytics.analytics_models.AnalyticsResult`
(distance / speed per track) into per-player performance: work rate, activity
level, fatigue, an overall 0..10 rating, a performance status and an
AI-style insight sentence. Metrics are normalized RELATIVE TO THE SQUAD in
this match (so a rating reflects standing among teammates, not absolute
units), via :mod:`analytics.rating_engine`. Phases 1-5 are untouched — this
only reads their output.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional

from analytics.analytics_models import AnalyticsResult, TrackAnalytics
from analytics.rating_engine import (
    activity_score,
    classify_activity,
    classify_fatigue,
    classify_work_rate,
    generate_insight,
    performance_status,
    player_rating,
    work_rate_score,
)
from utils.logger import get_logger

logger = get_logger("analytics.player_analytics")

# Speed (km/h) above which a frame counts as "actively moving" (jog+).
_ACTIVE_KMH = 6.0


@dataclass
class PlayerPerformance:
    """Per-player performance summary (basic + derived metrics + insight)."""

    track_id: int
    team_id: Optional[int]
    role: str
    total_distance_m: float
    avg_speed_kmh: float
    max_speed_kmh: float
    work_rate: str
    activity_level: str
    fatigue_level: str
    performance_status: str
    player_rating: float
    insight: str
    player_name: Optional[str] = None

    @property
    def player_display_name(self) -> str:
        return self.player_name or f"Player {self.track_id}"

    def to_dict(self) -> dict:
        data = {
            "track_id": int(self.track_id),
            "team_id": (None if self.team_id is None else int(self.team_id)),
            "role": self.role,
            "player_display_name": self.player_display_name,
            "total_distance_m": round(float(self.total_distance_m), 2),
            "avg_speed_kmh": round(float(self.avg_speed_kmh), 2),
            "max_speed_kmh": round(float(self.max_speed_kmh), 2),
            "work_rate": self.work_rate,
            "activity_level": self.activity_level,
            "fatigue_level": self.fatigue_level,
            "performance_status": self.performance_status,
            "player_rating": round(float(self.player_rating), 1),
            "insight": self.insight,
        }
        if self.player_name:
            data["player_name"] = self.player_name
        return data


def speed_drop(track: TrackAnalytics) -> float:
    """Fractional speed drop from the first half of the track to the second.

    Positive -> the player slowed down later (a fatigue signal); 0 if it held
    up or sped up. Uses the per-frame smoothed speeds.
    """
    series = sorted(track.frame_speed_kmh.items())
    if len(series) < 4:
        return 0.0
    mid = median(f for f, _ in series)
    first = [s for f, s in series if f <= mid]
    second = [s for f, s in series if f > mid]
    if not first or not second:
        return 0.0
    a, b = sum(first) / len(first), sum(second) / len(second)
    if a <= 0:
        return 0.0
    return max(0.0, (a - b) / a)


def consistency(track: TrackAnalytics) -> float:
    """Fraction of sampled frames the player was actively moving (>_ACTIVE_KMH)."""
    speeds = track.frame_speed_kmh.values()
    if not speeds:
        return 0.0
    active = sum(1 for s in speeds if s >= _ACTIVE_KMH)
    return active / len(track.frame_speed_kmh)


def compute_player_performance(
    track: TrackAnalytics,
    max_distance_m: float,
    max_avg_speed_kmh: float,
) -> PlayerPerformance:
    """Derive one player's performance, normalized against the squad maxima."""
    dist_norm = (track.total_distance_m / max_distance_m) if max_distance_m > 0 else 0.0
    speed_norm = (
        track.avg_speed_kmh / max_avg_speed_kmh if max_avg_speed_kmh > 0 else 0.0)
    cons = consistency(track)
    drop = speed_drop(track)

    act_s = activity_score(speed_norm, dist_norm)
    wr_s = work_rate_score(dist_norm, cons)
    work_rate = classify_work_rate(dist_norm, cons)
    activity = classify_activity(speed_norm, dist_norm)
    fatigue = classify_fatigue(dist_norm, drop, act_s)
    rating = player_rating(dist_norm, speed_norm, act_s, wr_s)
    status = performance_status(rating)
    insight = generate_insight(
        track.role, work_rate, activity, fatigue, rating, track.max_speed_kmh)

    return PlayerPerformance(
        track_id=track.track_id,
        team_id=track.team_id,
        role=track.role,
        total_distance_m=track.total_distance_m,
        avg_speed_kmh=track.avg_speed_kmh,
        max_speed_kmh=track.max_speed_kmh,
        work_rate=work_rate,
        activity_level=activity,
        fatigue_level=fatigue,
        performance_status=status,
        player_rating=rating,
        insight=insight,
        player_name=track.player_name,
    )


def compute_player_analytics(
    result: AnalyticsResult,
    roles: Optional[List[str]] = ("player", "goalkeeper"),
) -> List[PlayerPerformance]:
    """Per-player performance for every (player/goalkeeper) track in ``result``.

    Normalization is relative to the analysed squad: distance and average
    speed are scaled by the cohort maxima so ratings rank players within the
    match. Tracks flagged ``insufficient_data`` are skipped.
    """
    keep = set(roles or ())
    tracks = [
        t for t in result.tracks
        if (not keep or t.role in keep) and not t.insufficient_data
    ]
    if not tracks:
        return []
    max_dist = max(t.total_distance_m for t in tracks)
    max_speed = max(t.avg_speed_kmh for t in tracks)

    players = [compute_player_performance(t, max_dist, max_speed) for t in tracks]
    players.sort(key=lambda p: -p.player_rating)
    logger.info("Player performance computed for %d players", len(players))
    return players
